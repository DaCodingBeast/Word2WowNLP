import spacy
import csv

nlp = spacy.load("en_core_web_sm")

# Possibilities - found from spaCy Model
POS_TAGS = {tag: idx for idx, tag in enumerate(['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'])}
DEP_TAGS = {tag: idx for idx, tag in enumerate(nlp.get_pipe("parser").labels)}  # Dependency labels from the parser
ENT_TYPES = {label: idx for idx, label in enumerate(nlp.get_pipe("ner").labels)}  # Entity types from the NER component
TAG_TAGS = {tag: idx for idx, tag in enumerate(nlp.get_pipe("tagger").labels)}  # Detailed POS tags from the tagger

print("POS Tags:")
print(POS_TAGS)
print(f"Total POS tags: {len(POS_TAGS)}\n")

# Print DEP_TAGS and measure total
print("Dependency Tags:")
print(DEP_TAGS)
print(f"Total Dependency tags: {len(DEP_TAGS)}\n")

# Print ENT_TYPES and measure total
print("Entity Types:")
print(ENT_TYPES)
print(f"Total Entity types: {len(ENT_TYPES)}\n")

# Print TAG_TAGS and measure total
print("Detailed POS Tags:")
print(TAG_TAGS)
print(f"Total Detailed POS tags: {len(TAG_TAGS)}\n")


def create_dataset(paragraphs: list, results:dict):
    dataset = []
    for paragraphIndex in range(len(paragraphs)):
        
        WordIndex = 0
        doc = nlp(paragraphs[paragraphIndex])
        
        sentenceIndex = 0

        for sent in doc.sents:
            sentence_data = {"words": []}
            
            for word in sent:

                # Encode features as numbers
                # -1 default if not known
                pos_encoded = POS_TAGS.get(word.pos_, 18)
                dep_encoded = DEP_TAGS.get(word.dep_, 46)
                ent_type_encoded = ENT_TYPES.get(word.ent_type_, 19) if word.ent_type_ else 19
                detailed_pos_encoded = TAG_TAGS.get(word.tag_, 51)

                binaryResults = [0] * len(sent)

                possibleResult = results.get(paragraphIndex).get(WordIndex, 0.0)
                if(possibleResult != 0.0):
                    for index in possibleResult:
                        binaryResults[index] = 1.0
                    
                
                word_data = {
                    "group_id": paragraphIndex,
                    "index": WordIndex,
                    "word": word.text,                      # Word (raw)
                    "pos": pos_encoded,                     # Encoded POS tag
                    "detailed_pos": detailed_pos_encoded,   # Encoded Detailed POS tag
                    "dep": dep_encoded,                     # Encoded Dependency relation
                    "ent_type": ent_type_encoded,           # Encoded Named Entity type
                    "sent": sentenceIndex,                     # Sentence start index
                    "results": binaryResults
                }
                WordIndex +=1
                sentence_data["words"].append(word_data)

            sentenceIndex +=1
            dataset.append(sentence_data)
    
    return dataset

def export_to_csv(dataset, filename="dataset.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        writer.writerow([
            "group_id","index","word", "pos", "detailed_pos", "dep", "ent_type", "sent","results"])
        
        # Iterate through dataset
        for sentence in dataset:
            for word in sentence["words"]:
                row = [
                    word["group_id"],
                    word["index"],
                    word["word"],        # Word (raw)
                    word["pos"],         # Encoded POS tag
                    word["detailed_pos"],# Encoded Detailed POS tag
                    word["dep"],         # Encoded Dependency relation
                    word["ent_type"],    # Encoded Named entity type
                    word["sent"],        # Sentence start index
                    word["results"]
                ]
                writer.writerow(row)


