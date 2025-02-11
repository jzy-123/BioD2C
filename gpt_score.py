import openai
import pandas as pd
from tqdm import tqdm

client = openai.OpenAI(
    api_key="your api key",
    base_url="base_url"
)

def check_answer(question, pred, label):

    prompt = f"""
    You are a medical image analysis expert. Your task is to evaluate the predicted answer's rationality, accuracy, and similarity to the ground truth label.

    Question: {question}
    Predicted Answer: {pred}
    Ground Truth Label: {label}

    Please provide a score between 0 and 10, where 0 means the predicted answer is completely incorrect or unreasonable, 
    and 10 means the predicted answer is highly accurate, reasonable, and closely matches the ground truth label. 
    Provide only the score without any explanation.
    """

    messages = [{"role":"system","content":"You are an expert in medical image analysis and natural language processing."}]
    messages.append({"role":"user","content":prompt})
    try:
        res = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Error"

def process_csv(input_file, output_file):

    data = pd.read_csv(input_file)

    results = []
    scores = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        try:
            question = row["Question"]
            pred = row["Pred"]
            label = row["Label"]
            figure_path = row["Figure_path"]

            score = check_answer(question, pred, label)
            score = float(score)
            results.append({
                "Figure_path": figure_path,
                "Question": question,
                "Pred": pred,
                "Label": label,
                "gpt_score": score
            })
            scores.append(score)
        except Exception as e:
            print(f"Error processing row: {e}")

    print(sum(scores)/len(scores))
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    

input_file = "./rescsv/vqa.csv"  
output_file = "./rescsv/gpt_score.csv"  


process_csv(input_file, output_file)
