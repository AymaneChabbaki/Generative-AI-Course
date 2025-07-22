# Generative-AI-Course

# Generative AI for NET

## **Introduction to Machine Learning & AI**

**Artificial Intelligence** is a set of technologies that enable computers to perform human-like tasks:

- It performs **cognitive functions** we usually associate with human minds.
- **AI** is an *umbrella term* for any theory, computer system, or software developed to allow machines to perform tasks that normally require human intelligence.

<div style="text-align:center; margin: auto;">
<img style="height:80%; width:80%;" src="./img/image.png" /></div>
<div style="page-break-after: always;"></div>

### Machine Learning

Machine Learning is a form of artificial intelligence that is able to learn without explicit programming by humans.  
Some machine **learning algorithms** specialize in training themselves to detect patterns (**Deep Learning**).  
Uses:

- **Image Recognition**
- **Natural Language Processing**
- **Recommendation Systems**
- **Predictive Maintenance** (predict equipment failures before they occur)

### Introduction to Generative AI

Generative AI is artificial intelligence capable of creating new content using models trained on existing data (it generates text, images, audio, and video).  
Powered by advanced **Machine Learning Models** such as:

- *Transformers*
- *GANs* (**Generative Adversarial Networks**)
- Others

### Generative AI isn't

- Fully Autonomous
- Free from Bias
- Always accurate
- A replacement for humans

### Ethical Implications of AI

AI has the potential to unlock all kinds of opportunities (good or bad) for businesses, governments, and society:

- Non-discrimination
- Fairness
- Human Rights
- Environmental Impact
- Accountability
- Privacy and Data Protection

### Generative AI Best Practices

- Be aware of biases, inaccuracies, and ethical concerns.
- Always validate the AI-generated output, especially in critical applications.
- Ensure compliance with privacy laws, as generative models often use large datasets.
- Make it clear when content is AI-generated to avoid confusion or misinformation.
- Avoid using generative AI for malicious purposes, such as deep fakes or misleading information.

## Machine Learning Basics

**Training Set**:

- Used to train the Machine Learning Models.

**Validation Set**:

- Used to tune the model's hyperparameters and prevent overfitting.

**Test Set**:

- Used to evaluate the final performance of a Machine Learning Model.

### Types of Machine Learning

- **Supervised Learning** (labeled data)
- **Unsupervised Learning** (unlabeled data)
- **Semi-Supervised Learning** (labeled & unlabeled data)
- **Reinforcement Learning** (trial & error learning process with reward)

### Machine Learning Models

- **Generative Adversarial Networks** (GANs):
  - Used primarily for generating synthetic data.
- **Variational AutoEncoders** (VAEs):
  - Used for generating new data.
- **AutoRegressive Models** (AR):
  - Statistical model used for analyzing and predicting time series data.

### **Neural Networks**

Neural Networks are computational models designed to recognize patterns and make decisions based on data.

Training a Neural Network is the process of using training data to find the appropriate weights of the network for creating a good mapping of inputs and outputs  
<div style="text-align:center; margin: auto;">
<img style="height:60%; width:100%;" src="./img/image2.png" />
<img style="height:60%; width:100%;" src="./img/image3.png" />
</div>

## ML.NET

**ML.NET** is an open-source, cross-platform Machine Learning Framework for .NET:

- Allows you to apply your existing .NET skills and use the tools you're familiar with (like Visual Studio) to train **Machine Learning Models**.
- **Model Builder** is a **Graphical Visual Studio Extension** to train and deploy custom Machine Learning Models using ML.NET:
  - Uses **Automated Machine Learning** (AutoML) to find the *best model* for your data.
  - This automates the process of applying Machine Learning to data.

<div style="text-align:center; margin: auto;">
<img style="height:80%; width:100%;" src="./img/image4.png" />
</div>

### Data Preparation & Loading

The first step to train a Machine Learning Model is deciding which scenario and Machine Learning Task are the most appropriate for the solution:

- **Categorizing Data** (organize news articles by topic)
- **Predicting a Numerical Value** (estimate the price of a home)
- **Grouping Items with Similar Characteristics** (segment customers)
- **Classifying Images** (tag an image based on its contents)
- **Recommending Items**
- **Detecting Objects in Images** (detect pedestrians at an intersection)

The **scenarios** map to **Machine Learning Tasks**:

- A Machine Learning Task is the type of prediction/inference being made, based on the problem or question and the available data.
- With your scenario and training environment selected, it's time to load and prepare your data:
  - Choose your data source type.
  - Provide the location of your data.
  - Choose **Column purpose**:
    - You must define the purpose of certain columns.
    - In scenarios like data Classification and Value Prediction, you choose which of your columns is the one you want to predict (label).
    - All other columns that are not the label are used as features.
    - **Features** are **columns** used as inputs to **predict** the label.

Depending on your scenario, Model Builder supports loading data from the following sources:

- **Delimited files** (comma, semicolon, tab)
- Local/Remote **SQL Server Databases**
- **Images**

**Training** in Model Builder:

- Training consists of applying algorithms for the chosen scenario to your dataset to find the **best model**.
- To train **Machine Learning Models** in Model Builder, you only need to provide the time you want to train for:
  - Dataset -> `0 MB` to `10 MB`: 10 seconds
  - Dataset -> `10 MB` to `100 MB`: 10 minutes
  - Dataset -> `100 MB` to `500 MB`: 30 minutes
  - Dataset -> `500 MB` to `1 GB`: 60 minutes
  - Dataset -> `1 GB+`: 3+ hours
- Longer training periods allow Model Builder to explore more models with a wider range of settings.
- Model Builder uses **AutoML** to determine the transformations needed to prepare your data for training, select an algorithm, and tune the **hyperparameters** of the algorithm.
- By using evaluation metrics specific to the selected Machine Learning Task, Model Builder can determine which model performs best for your data.

## Generative AI

**Artificial Intelligence** (AI) imitates human behavior by using Machine Learning to interact with the environment and perform tasks without explicit instructions on what to output:

- **Generative AI** describes a category of AI capabilities that can create original content.
- Generative AI applications accept *natural language* input and return appropriate responses in natural language, images, or code.

**Generative AI** applications are powered by **Language Models**:

- These are specialized Machine Learning models used to perform **Natural Language Processing** (NLP) tasks.
- It's often more practical to use an existing foundation model and optionally fine-tune it with your own training data. Alternatively, you can train Language Models from scratch.
- **Language Models** are typically categorized into two types:
  - **Large Language Models** (LLMs):
    - Trained on vast amounts of text covering a broad range of general subject areas.
  - **Small Language Models** (SLMs):
    - Trained on smaller, more domain-specific datasets.

### Azure OpenAI Models

On **Microsoft Azure**, foundation models are available through the **Azure OpenAI service** and the **Model Catalog** (a curated collection of pre-trained models).  
Using models from the Azure OpenAI service provides the added benefits of a secure and scalable **Azure Cloud Platform**, where the models are hosted and managed.

## SETUP PREDICTIVE MODEL

### Visual Studio 2022

- **Create a new Project**
  - **Console App**:
    - Create a new Console App project, e.g., `MLProject.Console`
    - Inside the project, go to `Add -> Machine Learning Model...`
    - Select **Machine Learning Model (ML.NET)** (default option)
    - Click **Add**: this will generate a new file in the solution named `xxx.mbconfig`

- **Model Setup**
  - **Scenario**: Data Classification
  - **Environment**: Local (CPU)
  - **Data (for training)**:
    - Add the downloaded `.csv` file (training dataset)
    - Column to predict (label): **Machine failure**
    - Click **Start Training**
    - **Advanced training options**:
      - Used to retrain the model
      - Allows tuning **hyperparameters** to avoid **overfitting** or **underfitting**

  - **Evaluate**:
    - Displays model evaluation metrics such as *accuracy*, *precision*, *recall*, and *F1-score*
    - Allows comparison between different models to choose the best one

  - **Consume**:
    - Automatically generates C# code to use the trained model
    - Creates a class to load the model and make predictions on new data

---

### Visual Studio Code (ML.NET CLI)

- **Install ML.NET CLI Tool**

  - **Linux**
    ```bash
    dotnet tool install -g mlnet-linux-x64      # for 64-bit systems  
    dotnet tool install -g mlnet-linux-arm64    # for ARM64 systems
    ```

  - **macOS**
    ```bash
    dotnet tool install -g mlnet-osx-x64        # for 64-bit systems  
    dotnet tool install -g mlnet-osx-arm64      # for ARM64 systems
    ```

  - **Windows**
    ```bash
    dotnet tool install --global mlnet-win-x64     # for 64-bit systems  
    dotnet tool install --global mlnet-win-arm64   # for ARM64 systems
    ```

- **Train the model via terminal**
  ```bash
  mlnet classification --dataset "data.csv" --label-col 8 --has-header true --name PredictiveModel --train-time 10
  ```

  - `--dataset`: path to the dataset to be used
  - `--label-col`: index of the column to be predicted (target/label)
  - `--has-header`: indicates whether the dataset includes a header row
  - `--name`: name for the generated ML model and related assets
  - `--train-time`: number of seconds for ML.NET to explore and train models

[ML.NET Tutorial](https://dotnet.microsoft.com/en-us/learn/ml-dotnet/get-started-tutorial/consume)
[ML.NET Documentation](https://learn.microsoft.com/en-us/dotnet/machine-learning/)
