# Example Command
### Back-Translate
```bash
# en -> ru
python backtranslate.py -dn 4 \
    -bs 128 \
    -m enru \
    -o ../data/backtranslation \
    -p ../data/classification/skoltech-jigsaw/train.txt \
        ../data/classification/skoltech-jigsaw/valid.txt \
        ../data/classification/skoltech-jigsaw/test.txt


# ru -> en
python backtranslate.py -dn 4 \
    -bs 64 \
    -m ruen \
    -o ../data/backtranslation \
    -p ../data/backtranslation/enru_train.txt \
        ../data/backtranslation/enru_valid.txt \
        ../data/backtranslation/enru_test.txt 
```



### Paraphraser
```bash
# train few-shot gpt2
python paraphrase.py \
    -mp ./model/paraphrase_new.pt \
    --data paraphrase_ref.csv \
    -m paraphrase \
    -dn 3 \
    -b 32 \
    -e 5

# generate paraphrase
python paraphrase-generator.py \
    -dn 4 \
    -mp ./model/paraphrase_new.pth \
    --data ./data/paraphrase/train_gen_pair.txt \
    -nb 5 \
    -o ./data/paraphrase/train_gen_pair_new.txt \
    -sc ori \
    --decoding-method self \
    --topk 10 \
    --topp 0.66
```