Prerequisites
    Python 3.7+
    OpenAI API key
    Google Serper API key
    Installation and Local Setup
Clone the repository:
    git clone 
    cd PROMPTTECHNIQUE
Set up a virtual environment:

    On macOS and Linux:
        python3 -m venv venv
        source venv/bin/activate
    On Windows:
        python -m venv venv
        .\venv\Scripts\activate

Install the required packages:
    pip install -r requirements.txt

Create a .env file in the project root and add your API keys:
OPENAI_API_KEY=your_openai_api_key_here

Run the application:
python zeroshot.py