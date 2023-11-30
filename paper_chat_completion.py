# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

import json

import spacy

from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    paper_dict_path: str
    phrase: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    with open(paper_dict_path) as json_file:
        paper_dict = json.load(json_file)

    dialogs: List[Dialog] = [
        [
            {"role": "user", "content": f"Tell me a name of paper related to {phrase}. Just the title and authors. Start with Title: and Authors:"},
        ],]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    
    for dialog, result in zip(dialogs, results):
        #for msg in dialog:
            #print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        #print(
            #f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        #)
        sentences = result['generation']['content'].split('\n')
        for sentence in sentences:
            if sentence.startswith('Title:'):
                title = sentence.split(':')[1:]
                new_title = ':'.join(title)
            elif sentence.startswith('Authors:'):
                authors = sentence.split(':')[1]
                authors = authors.split(',')
                authors = [author.strip() for author in authors]
                try:
                    authors[-1] = authors[-1].replace('and ','')
                except:
                    authors = authors
        while new_title[0] == '"' or new_title[0] == ' ':
            new_title = new_title[1:]
        while new_title[-1] == '"' or new_title[-1] == ' ':
            new_title = new_title[:-1]
        flag = 0
        sim_score = {}
        print("\n==================================\n")
        print("looking for match in the dict...")   
        for paper in tqdm(paper_dict):
            if new_title in paper_dict[paper]["title"]:
                flag = 1
                break
            similarity = nlp(new_title).similarity(nlp(paper_dict[paper]["title"]))
            if len(sim_score) < 10:
                sim_score[paper] = similarity
            else:
                if similarity > min(sim_score.values()):
                    sim_score[paper] = similarity
                    del sim_score[min(sim_score, key=sim_score.get)]
        print("\n==================================\n")
        if flag == 0:
            print(f"Generated Title: {new_title}")
            print(f"Generated Authors: {authors}")
            print("No match found in the dict. Similar papers are:")
            with open('count/paper_count_wrong.txt', 'r') as f:
                count = int(f.read())
            count += 1
            with open('count/paper_count_wrong.txt', 'w') as f:
                f.write(str(count))
            for paper in sim_score:
                paper_title = paper_dict[paper]["title"]
                print(f"{paper_title}: {sim_score[paper]}")
        else:
            with open('paper_count_year_right.txt', 'r') as f:
                count = int(f.read())
            count += 1
            with open('count/paper_count_year_right.txt', 'w') as f:
                f.write(str(count))
            print(f"Generated Title: {new_title}")
            print(f"Generated Authors: {authors}\n")
            print(f"Match found in the dict: {paper}")
            print(f"Title: {paper_dict[paper]['title']}")
            print(f"Authors: {paper_dict[paper]['authors']}")

        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
