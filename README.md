# SciHal
## Quick Start
1. Git clone llama 2 [here](https://github.com/facebookresearch/llama.git)
2. Place all the files and folders from this repository in the llama folder.
3. Download llama 2 using `./download.sh`.
4. Run the following code to run phrases using the keyword
   ```bash
   chmod +x run_phrases.sh
   ./run_phrases.sh papers.txt --key {your semantic scholar key}
   ```
5. Run the following code to run phrases using Author names
   ```bash
   chmod +x run_phrases.sh
   ./run_phrases.sh papers.txt --key {your semantic scholar key} --rev_author_dict_path {path to reverse author directory}
   ```
6. To check the existence of the paper in the dataset, run the following command:
   ```bash
   chmod +x run_dataset_search.sh
   ./run_dataset_search.sh papers.txt --paper_dict_path {path to paper directory}
   ```
This will update the number of correct and wrong predictions based on the year, authors, and verification of the response. 
