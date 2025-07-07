import csv
import os

class CSVWriter:

    def write_to_csv(self, filepath, data, fieldnames):
        """
        Write a single row of data to a CSV file.

        Args:
            filepath (str): Path to the CSV file.
            data (dict): A dictionary representing the row to write.
            fieldnames (list): List of column names (keys of the data dict).
        """
        file_exists = os.path.isfile(filepath)

        with open(filepath, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)

    def parse_question_csv(self, filepath):
        """
        Parse a CSV file and return a list of dictionaries.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            list: A list of dictionaries representing each row in the CSV.
        """
        data = []
        with open(filepath, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for row in reader:
                data.append(row)
        return data
    
csv_writer = CSVWriter()