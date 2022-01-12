import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
# labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
# # the forward function automatically creates the correct decoder_input_ids
# pre = model(input_ids=input_ids, labels=labels)
# print(pre.shape)


# def mask_ind(tokens):
#     tokens[0] = 'a'


# def main():
#     tokens = ["asdfasdf",
#             "asdfasdfasdf"]

#     mask_ind(tokens)
#     for str in tokens:
#         print(str)




# if __name__ == '__main__':
#     main()


# Load T5 model and T5 tokenizer (for quicker loading)
tokenizer = AutoTokenizer.from_pretrained(k_model)
model = T5ForConditionalGeneration.from_pretrained(k_model)
model = nn.DataParallel(model,device_ids=[0,1])

def forward(model, device, batch):
    src_ids = batch["source_ids"].to(device, dtype=torch.long)
    src_mask = batch["source_mask"].to(device, dtype=torch.long)
    tgt_ids = batch["target_ids"].to(device, dtype=torch.long)

    # Pad ids (pad=0) are set to -100, which means ignore for loss calculation
    tgt_ids[tgt_ids[: ,:] == 0 ] = -100
    label_ids = tgt_ids.to(device)

    # NOTE: when we call model() with labels, they will be
    # - automatically right shifted by 1 (for teacher forcing)
    # - prepended by BOS=Beginning of sequence which is a PAD token
    # - any token that was -100 will be masked_fill_ to <pad> for teacher forcing return_dict means return as a dictionary
    
    out_dict = model(src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True)
    loss, logits = out_dict['loss'], out_dict['logits']
    return loss, logits

def pipeline(tokenizer, model):
    # Set random seeds
    util.set_seed(k_seed)

    # Set device and GPU list
    device, gpu_ids = util.get_available_devices()

    # Load training dataset and validation dataset
    train_loader, dev_loader = get_dataloaders(tokenizer=tokenizer, batch_size=k_batch_size, num_train=k_num_train, num_val=k_num_val, data_dir=k_data_dir, num_workers=k_num_workers)

    # Reset in case we used the -1 flag for all
    num_train   = len(train_loader.dataset)
    num_val     = len(dev_loader.dataset)
    total_steps = (num_train // k_batch_size) * k_epochs
    total_train = num_train * k_epochs

    model.to(device)


    optimizer = AdamW(model.parameters(), lr=k_lr, eps=k_adam_eps)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=k_warmup_steps, num_training_steps=total_steps)

    logger.info(f'device: {device}\n'
                f'gpu_ids: {gpu_ids}\n'
                f'total_steps: {total_steps}\n'
                f'total_train (num_t * epoch): {total_train}\n')

    epoch = 0       # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)

    while epoch < k_epochs:
        epoch += 1

        ### Training
        model.train()
        logger.info(f'Training at step {step}...')

        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                real_batch_size = len(batch["source_ids"])

                loss, logits = forward(model, device, batch)
                loss_val = loss.mean().item()      # get the item since loss is a tensor

                # Backward
                optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), k_max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Log info
                step += real_batch_size
                progress_bar.update(real_batch_size)
                progress_bar.set_postfix(epoch=epoch, loss=loss_val)

                tbx.add_scalar('train/loss', loss_val, step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'],step)

        ### Evaluation
        logger.info(f'Evaluating at step {step}...')

        # For parallel model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        model.eval()

        # See how the model is doing with exact match on tokens
        pred_list_all = []                  # Accumulate for saving; list; one list per epoch
        pred_list_correct = []
        loss_meter = util.AverageMeter()    # NLL (default metric for model) (reset each time)

        # Set up two count variables
        total_matches_no_eos_ct = 0
        total_matches_with_eos_ct = 0

        with torch.no_grad(), tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(dev_loader):
                real_batch_size = len(batch["source_ids"])

                # Evaluation for loss fcn
                loss, logits = forward(model, device, batch)
                loss_meter.update(loss.mean().item(), real_batch_size)

                # Predict/Generate for token matches
                src_ids = batch["source_ids"].to(device, dtype=torch.long)
                src_mask = batch["source_mask"].to(device, dtype=torch.long)
                tgt_ids = batch["target_ids"].to(device, dtype=torch.long)

                # Tweak the generation params. See huggingface details for generate
                # Batch generate
                generated_ids = model.generate(src_ids, attention_mask=src_mask)       

                # Collect some stats
                total_matches_no_eos, total_matches_with_eos, correct_indices = util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
                total_matches_no_eos_ct += total_matches_no_eos
                total_matches_with_eos_ct += total_matches_with_eos

                # Save for qualitative analysis
                orig_text_input, orig_text_output = batch["source_text"], batch["target_text"]
                # todo: this could break once skip_special_tokens is fixed
                outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                preds = list(zip(orig_text_input, orig_text_output, outputs_decoded))
                pred_list_all.extend(preds)

                # We also store only the correct indices
                for idx in correct_indices.tolist():    # tensor to list; these are the valid indices
                    pred_list_correct.append(preds[idx[0]])     # each item was a list of one element

                # Log info
                progress_bar.update(real_batch_size)
                progress_bar.set_postfix(NLL=loss_meter.avg)

        # Save predictions for qualititative analysis
        util.save_preds(pred_list_all, record_dir)
        util.save_preds(pred_list_correct, record_dir, file_name="preds_correct.csv")
        results_list = [('NLL', loss_meter.avg),
                        ('exact_match_with_eos', total_matches_with_eos_ct),
                        ('exact_match_no_eos', total_matches_no_eos_ct)]
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        logger.info(f'Dev {results_str}')

        # Log to TensorBoard
        for k, v in results.items():
            tbx.add_scalar(f'dev/{k}', v, step)

        util.visualize(tbx, pred_dict=pred_list_all, step=step,split='dev', num_visuals=3)    

pipeline(tokenizer, model)
