# ford_car_sentiment_analysis# 

## **Problem Statement:**
Analyze and extract insights from Ford car reviews using LangChain. This includes performing sentiment analysis and identifying key information such as pros, cons, and specific features mentioned in the reviews.

---

## **Steps to Solve:**

### **Step 1: Load the Dataset**
1. Download the dataset `ford_car_reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews).
2. Limit the dataset to the first 25 records using:
   ```python
   df = pd.read_csv('ford_car_reviews.csv')
   df = df.head(25)
   ```

---

### **Step 2: Define the Sentiment Analysis Task**
- Use LangChain to classify the sentiment of each review into **positive**, **neutral**, or **negative**.
- Example prompt:
  ```
  "Given the following car review, classify the sentiment as positive, neutral, or negative:
  Review: '{review_text}'
  Output: Sentiment"
  ```

#### **LangChain Integration**
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Initialize LLM
llm = OpenAI(temperature=0.0)

# Define sentiment analysis prompt template
sentiment_prompt = PromptTemplate(
    input_variables=["review_text"],
    template="Given the following car review, classify the sentiment as positive, neutral, or negative:\nReview: '{review_text}'\nOutput: Sentiment"
)

def analyze_sentiment(review):
    prompt = sentiment_prompt.format(review_text=review)
    sentiment = llm(prompt)
    return sentiment.strip()

df["Sentiment"] = df["review_text"].apply(analyze_sentiment)
```

---

### **Step 3: Key Insights Extraction**
- Extract **pros**, **cons**, and **specific features liked/disliked** from reviews.
- Example prompts:
  1. Extracting pros and cons:
     ```
     "What are the pros and cons of the vehicle described in the following review?\nReview: '{review_text}'\nOutput: Pros: ...\nCons: ..."
     ```
  2. Identifying liked and disliked features:
     ```
     "What specific features of the vehicle does the reviewer like or dislike?\nReview: '{review_text}'\nOutput: Liked Features: ...\nDisliked Features: ..."
     ```

#### **LangChain Integration for Insights Extraction**
```python
# Define prompts for insights extraction
pros_cons_prompt = PromptTemplate(
    input_variables=["review_text"],
    template="What are the pros and cons of the vehicle described in the following review?\nReview: '{review_text}'\nOutput: Pros: ...\nCons: ..."
)

features_prompt = PromptTemplate(
    input_variables=["review_text"],
    template="What specific features of the vehicle does the reviewer like or dislike?\nReview: '{review_text}'\nOutput: Liked Features: ...\nDisliked Features: ..."
)

def extract_pros_cons(review):
    prompt = pros_cons_prompt.format(review_text=review)
    result = llm(prompt)
    pros, cons = result.split("Cons:")
    return pros.strip().replace("Pros:", ""), cons.strip()

def extract_features(review):
    prompt = features_prompt.format(review_text=review)
    result = llm(prompt)
    liked, disliked = result.split("Disliked Features:")
    return liked.strip().replace("Liked Features:", ""), disliked.strip()

df["Pros"], df["Cons"] = zip(*df["review_text"].apply(extract_pros_cons))
df["Liked_Features"], df["Disliked_Features"] = zip(*df["review_text"].apply(extract_features))
```

---

### **Step 4: Update the DataFrame with New Information**
- Add the following columns to the DataFrame:
  - `Sentiment`
  - `Pros`
  - `Cons`
  - `Liked_Features`
  - `Disliked_Features`
- Save the updated DataFrame for further analysis.

#### **Save Updated DataFrame**
```python
# Save to a new CSV file
updated_file = "updated_ford_car_reviews.csv"
df.to_csv(updated_file, index=False)
print(f"Updated data saved to {updated_file}")
```

---

## **Expected Outputs:**
- A structured DataFrame with the following additional columns:
  - **Sentiment**: Positive, Neutral, or Negative.
  - **Pros**: Key positive aspects mentioned.
  - **Cons**: Key negative aspects mentioned.
  - **Liked_Features**: Specific features praised by the reviewer.
  - **Disliked_Features**: Specific features criticized by the reviewer.

---

## **Extensions:**
1. **Visualization**:
   - Create visualizations of the sentiment distribution.
   - Highlight the most frequently mentioned pros and cons.
2. **Improved Analysis**:
   - Use a fine-tuned LLM for better accuracy in insights extraction.
3. **Scalability**:
   - Integrate the pipeline with a cloud service for real-time analysis of incoming reviews.
