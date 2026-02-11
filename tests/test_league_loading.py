import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rnad import LeagueConfig

def test_league_loading():
    student_csv = "train_data/student.csv"
    teacher_csv = "train_data/teacher.csv"
    
    if not os.path.exists(student_csv) or not os.path.exists(teacher_csv):
        print("CSV files not found. Skipping test.")
        return

    print(f"Loading leagues from {student_csv} and {teacher_csv}...")
    config = LeagueConfig.from_csv(student_csv, teacher_csv)
    
    print(f"Student decks: {len(config.student_decks)}")
    print(f"Teacher decks: {len(config.teacher_decks)}")
    
    assert len(config.student_decks) > 0
    assert len(config.teacher_decks) > 0
    
    # Check if first few decks exist
    assert os.path.exists(config.student_decks[0])
    assert os.path.exists(config.teacher_decks[0])
    
    print("Sampling 10 decks...")
    d1, d2 = config.sample_decks(10)
    
    print("Sampled Player 1 (Student):")
    for d in d1:
        print(f"  {os.path.basename(d)}")
        assert d in config.student_decks
        
    print("Sampled Player 2 (Teacher):")
    for d in d2:
        print(f"  {os.path.basename(d)}")
        assert d in config.teacher_decks
        
    print("Test passed!")

if __name__ == "__main__":
    test_league_loading()
