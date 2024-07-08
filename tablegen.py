def encode_spaces(url):
    return url.replace(' ', '%20')

def generate_markdown_table(papers):
    table_header = "| Paper Name | ArXiv Link |\n|------------|------------|\n"
    table_rows = ""
    
    for paper in papers:
        paper_name = paper['name']
        arxiv_link = paper['arxiv']
        repo_url = encode_spaces(paper['repo'])
        table_rows += f"| [{paper_name}](./{repo_url}) | [{arxiv_link if 'arxiv' in arxiv_link else 'paper'}]({arxiv_link}) |\n"
    
    return table_header + table_rows

def main():
    papers = []
    
    while True:
        name = input("Enter the paper name (or type 'done' to finish): ")
        if name.lower() == 'done':
            break
        
        arxiv = input("Enter the ArXiv URL or link: ")
        repo = input("Enter the repository URL: ")
        
        papers.append({
            'name': name,
            'arxiv': arxiv,
            'repo': repo
        })
    
    markdown_table = generate_markdown_table(papers)
    print("\nGenerated Markdown Table:\n")
    print(markdown_table)

if __name__ == "__main__":
    main()
	# Paper

# (or type 'done' to finish): Improving Language Understanding by Generative Pre-Training (GPT2)
# Enter the ArXiv URL or link: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
# Enter the repository URL: GPT2
# Enter the paper name (or type 'done' to finish): done