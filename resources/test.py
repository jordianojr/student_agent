import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test():
    df = pd.read_csv('manual_options_45MC.csv', sep=',')
    question_dict = {}
    option_id = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    
    for row in df.itertuples():
        if row.qn_id not in question_dict:
            question_dict[row.qn_id] = {
                'qn_week': str(row.week),  # Convert week to string
                'qn_stem': row.stem,  # Renamed for clarity
                'options': [],
                'qn_options': row.stem,
                'correct_option': ""  # This will build the full question with options
            }
        
        if row.is_answer:
            # If this row is marked as the answer, store the option letter
            question_dict[row.qn_id]['correct_option'] = option_id[len(question_dict[row.qn_id]['options'])]

        # Get the current option letter based on how many options we already have
        current_option_letter = option_id[len(question_dict[row.qn_id]['options'])]
        
        # Add to the options list
        question_dict[row.qn_id]['options'].append(
            current_option_letter
        )
        
        # Build the complete question string with all options so far
        if len(question_dict[row.qn_id]['options']) == 1:
            # First option - start with the stem
            question_dict[row.qn_id]['qn_options'] = row.stem + ' ' + current_option_letter + '. ' + row.option
        else:
            # Subsequent options - append to existing
            question_dict[row.qn_id]['qn_options'] += ' ' + current_option_letter + '. ' + row.option
    
    # Connect to MongoDB Atlas
    client = MongoClient(os.getenv('MONGODB_URL'))
    db = client['agents_db']  # Replace with your database name
    collection = db['questions_manual']
    
    # Convert dictionary to list of documents
    documents = [{'qn_id': k, **v} for k, v in question_dict.items()]
    
    # Insert documents into MongoDB
    if documents:
        collection.insert_many(documents)
        print(f"Successfully uploaded {len(documents)} questions to MongoDB")
    
    return question_dict

if __name__ == "__main__":
    test()