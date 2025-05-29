# import pandas as pd

# # Read the CSV file
# df = pd.read_csv('/raid/ganesh/pdadiga/kritarth/chriss/outputs-agents/qc2/llama33/QC2-3/wer.csv')

# # Calculate average WER by summing all WER values and dividing by number of rows
# avg_wer = df['WER'].mean()

# print(f"Average WER: {avg_wer:.4f}")








# import pandas as pd

# # Read the CSV file
# df = pd.read_csv('/raid/ganesh/pdadiga/kritarth/chriss/outputs-agents/qc2/llama33/QC2-3/english_word_count.csv')

# # Calculate sum of values in 'status' column
# status_sum = df['english_word_count'].sum()

# print(f"english_word_detected:  {status_sum}")











# import pandas as pd

# def count_language_occurrences(csv_file_path):
#     try:
#         # Read the CSV file into a DataFrame
#         df = pd.read_csv(csv_file_path)
        
#         # Count occurrences of each unique value in the 'Evaluation_Comment' column
#         language_counts = df['Evaluation_Comment'].value_counts()
        
#         # Format the output as a single line
#         output_line = " | ".join([f"{language}: {count}" for language, count in language_counts.items()])
        
#         # Print the output in a single line
#         print("LLM score counts (single line):")
#         print(output_line)
        
#         # Return the counts for potential further use
#         return language_counts

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# # Example usage
# if __name__ == "__main__":
#     # Replace with your CSV file path
#     csv_file_path = "/raid/ganesh/pdadiga/kritarth/chriss/outputs-agents/qc2/llama33/QC2-3/llm_scores.csv"
#     count_language_occurrences(csv_file_path)





# import pandas as pd

# def count_language_occurrences(csv_file_path):
#     try:
#         # Read the CSV file into a DataFrame
#         df = pd.read_csv(csv_file_path)
        
#         # Count occurrences of each unique value in the 'language' column
#         language_counts = df['Is_Devanagari'].value_counts()
        
#         # Display the counts
#         print("Language occurrence counts:")
#         for language, count in language_counts.items():
#             print(f"{language}: {count}")
        
#         # Return the counts dictionary for potential further use
#         return language_counts
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# # Example usage
# if __name__ == "__main__":
#     # Replace with your CSV file path
#     csv_file_path = "/raid/ganesh/pdadiga/kritarth/chriss/outputs-agents/qc2/gpt4.1mini/QC2-3/language_verification.csv"
#     count_language_occurrences(csv_file_path)








import pandas as pd

def count_language_occurrences(csv_file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        
        # Count occurrences of each unique value in the 'domain' column
        language_counts = df['domain'].value_counts()
        
        # Get top 5 most frequent items
        top_5_languages = language_counts.head(5)

        # Format and print the output in a single line
        output_line = " | ".join([f"{language}: {count}" for language, count in top_5_languages.items()])
        print("Top 5 language occurrence counts (single line):")
        print(output_line)

        # Return the top 5 counts
        return top_5_languages

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file_path = "/raid/ganesh/pdadiga/kritarth/chriss/outputs-agents/qc2/llama33/QC2-3/domain_check.csv"
    count_language_occurrences(csv_file_path)































# import pandas as pd
# import re

# # Load CSV file
# df = pd.read_csv("/raid/ganesh/pdadiga/kritarth/chriss/outputs-agents/qc2/llama33/QC2-3/normalized_list.csv")  # Replace with your actual file path

# # Ensure the column exists
# if 'normalized_transcripts' not in df.columns:
#     raise ValueError("The column 'normalized_transcripts' does not exist in the CSV file.")

# # Initialize counters
# tag_count = 0
# square_bracket_count = 0
# special_char_count = 0
# english_word_count = 0

# # Patterns
# html_tag_pattern = re.compile(r'</?[^>]+>')              # HTML-like tags
# square_bracket_pattern = re.compile(r'[\[\]]')           # [ or ]
# special_char_pattern = re.compile(r'[^\w\s\u0900-\u097F]')  # Not word, space, or Hindi char
# english_word_pattern = re.compile(r'\b[a-zA-Z]+\b')      # English words

# # Loop through each transcription
# for text in df['normalized_transcripts'].dropna():
#     tag_count += len(html_tag_pattern.findall(text))
#     square_bracket_count += len(square_bracket_pattern.findall(text))
#     special_char_count += len(special_char_pattern.findall(text))
#     english_word_count += len(english_word_pattern.findall(text))

# # Print results
# print(f"HTML-like tags: {tag_count}, Square brackets: {square_bracket_count}, Special characters: {special_char_count}, English words: {english_word_count}")

