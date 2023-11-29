# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

import json

from tqdm import tqdm

import requests

#nlp = spacy.load("en_core_web_sm")

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    key: str,
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

    dialogs: List[Dialog] = [
        [
            {"role": "user", "content": f"{phrase}"},
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
            elif sentence.startswith('Year:'):
                year = sentence.split(':')[1]
                year = year.strip()
        while new_title[0] == '"' or new_title[0] == ' ':
            new_title = new_title[1:]
        while new_title[-1] == '"' or new_title[-1] == ' ':
            new_title = new_title[:-1]
        print("Generated Title: ", new_title)
        print("Generated Authors: ", authors)
        print("Generated Year: ", year)
        print("\n==================================\n")
        response = requests.get("https://api.semanticscholar.org/graph/v1/paper/autocomplete?query=" + new_title.replace(' ','%20'), headers={'x-api-key':key}).json()
        try:
            if len(response['matches']) == 0:
                print("No such paper found....LLM is halucinating\n")
                response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search?query=" + new_title.replace(' ','%20') + "&fields=title,authors,year" ,headers={'x-api-key':key}).json()
                recom_paper = {}
                for paper in response['data']:
                    recom_paper[paper['paperId']] = {}
                    recom_paper[paper['paperId']]['title'] = paper['title']
                    recom_paper[paper['paperId']]['authors'] = []
                    for author in paper['authors']:
                        recom_paper[paper['paperId']]['authors'].append(author['name'])
                    recom_paper[paper['paperId']]['year'] = paper['year']
                print("Here are some recommendations: ")
                print("\n==================================\n")
                for paper in recom_paper:
                    print("Title: ", recom_paper[paper]['title'])
                    print("Authors: ", recom_paper[paper]['authors'])
                    print("Year: ", recom_paper[paper]['year'])
                    print("\n==================================\n")
                with open('paper_count_wrong.txt', 'r') as f:
                    count = int(f.read())
                count += 1
                with open('paper_count_wrong.txt', 'w') as f:
                    f.write(str(count))
            else:
                for paper in response['matches']:
                    if paper['title'].lower() == new_title.lower():
                        flag = 1
                        original_title = paper['title']
                        paper_id = paper['id']
                        response = requests.get("https://api.semanticscholar.org/graph/v1/paper/" + paper_id +"?fields=year,authors", headers={'x-api-key':key}).json()
                        original_authors = []
                        for author in response['authors']:
                            original_authors.append(author['name'])
                        original_year = response['year']
                        break
                print("Paper Verified by Semantic Scholar")
                print("Paper ID: ", paper_id)
                print("Paper Title: ", new_title)
                print("Paper Authors: ", original_authors)
                print("Paper Year: ", original_year)

                if original_year != year:
                    with open('paper_count_year_wrong.txt', 'r') as f:
                        count = int(f.read())
                    count += 1
                    with open('paper_count_year_wrong.txt', 'w') as f:
                        f.write(str(count))
                else:
                    with open('paper_count_year_right.txt', 'r') as f:
                        count = int(f.read())
                    count += 1
                    with open('paper_count_year_right.txt', 'w') as f:
                        f.write(str(count))
            
                print("\n==================================\n")
        except:
            with open('paper_count_wrong.txt', 'r') as f:
                count = int(f.read())
            count += 1
            with open('paper_count_wrong.txt', 'w') as f:
                f.write(str(count))
            
            print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
