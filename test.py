import random
# from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
# labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
# # the forward function automatically creates the correct decoder_input_ids
# pre = model(input_ids=input_ids, labels=labels)
# print(pre.shape)


def mask_ind(tokens):
    tokens[0] = 'a'


def main():
    tokens = ["asdfasdf",
            "asdfasdfasdf"]

    mask_ind(tokens)
    for str in tokens:
        print(str)

if __name__ == '__main__':
    main()