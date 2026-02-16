import nltk
import subprocess
import sys

def fix_textblob():
    """Fix TextBlob missing corpus error"""
    print("Fixing TextBlob missing corpus error...")
    
    # Download required NLTK data
    required_data = [
        'punkt',
        'averaged_perceptron_tagger',
        'brown',
        'wordnet',
        'vader_lexicon'
    ]
    
    for data in required_data:
        try:
            print(f"Downloading {data}...")
            nltk.download(data, quiet=False)
        except Exception as e:
            print(f"Error downloading {data}: {e}")
    
    # Try textblob download command
    try:
        print("\nRunning textblob download command...")
        subprocess.run([sys.executable, "-m", "textblob.download_corpora"], check=True)
    except:
        pass
    
    print("\n✅ Fix complete! Try running the app again.")

if __name__ == "__main__":
    fix_textblob()