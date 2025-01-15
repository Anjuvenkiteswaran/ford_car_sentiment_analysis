# ford_car_sentiment_analysis# 

## **Problem Statement:**
Analyze and extract insights from Ford car reviews using LangChain. This includes performing sentiment analysis and identifying key information such as pros, cons, and specific features mentioned in the reviews.

---

## **Steps to Solve:**

### **Step 1: Load the Dataset**
1. Download the dataset `ford_car_reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews).
2. Limit the dataset to the first 25 records using:
   ```python


---

### **Step 2: Define the Sentiment Analysis Task**
- Use LangChain to classify the sentiment of each review into **positive**, **neutral**, or **negative**.



#### **LangChain Integration**
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Initialize LLM
llm = OpenAI(temperature=0.0)



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
#

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


---

## **Expected Outputs:**
- A structured DataFrame with the following additional columns:
  - **Sentiment**: Positive, Neutral, or Negative.
  - **Pros**: Key positive aspects mentioned.
  - **Cons**: Key negative aspects mentioned.
  - **Liked_Features**: Specific features praised by the reviewer.
  - **Disliked_Features**: Specific features criticized by the reviewer.


