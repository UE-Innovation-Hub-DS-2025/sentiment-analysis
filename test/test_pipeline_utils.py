import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
from unittest.mock import patch, mock_open, MagicMock
from src.pipeline_utils import extract_text_from_folder


class TestExtractTextFromFolder(unittest.TestCase):
    @patch('os.listdir')
    @patch('os.path.isfile')
    @patch('builtins.open', new_callable=mock_open, read_data='sample review text')
    def test_extract_text_from_folder(self, mock_file, mock_isfile, mock_listdir):
        # Setup mocks
        mock_listdir.return_value = ['1_8.txt', '2_3.txt']
        mock_isfile.return_value = True
        folder_path = '/fake/folder'
        label = 'positive'
        # Call function
        result = extract_text_from_folder(folder_path, label)
        # Check results
        expected = [
            ('positive', 'sample review text', 8),
            ('positive', 'sample review text', 3)
        ]
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main() 