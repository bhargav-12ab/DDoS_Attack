# DDoS-Attack

**Project Overview** - This project focuses on detecting Distributed Denial of Service (DDoS) attacks in Software-Defined Networks (SDN) using machine learning models. The dataset used for this project contains both benign and malicious traffic (TCP SYN Flood, UDP Flood, ICMP Flood attacks). The goal is to classify traffic as normal (benign) or malicious using classical machine learning models such as Decision Tree, Naive Bayes, Support Vector Machine (SVC), and Random Forest.

**Dataset Description** - The dataset was generated using the Mininet emulator and includes a mix of benign and malicious traffic. It consists of 104,345 rows and 23 columns.

**Key Details** - Target Variable: label
1: Malicious Traffic
0: Benign Traffic

**Features** - Categorical Features: 3 (e.g., Switch ID, Source IP, Destination IP), Numeric Features: 20 (e.g., Packet Count, Byte Count, Port Number)

**Traffic Types** - Benign Traffic, TCP SYN Flood, UDP Flood, ICMP Flood

**Dataset Link** Kaggle DDoS SDN Dataset

**Features** - Switch ID, Packet Count, Byte Count, Duration, Source IP, Destination IP, Port Number, Protocol Type, Flow Duration...and more (23 total features).

**Machine Learning Models** - The following machine learning models were implemented to classify network traffic:
1. Decision Tree
2. Random Forest
3. Support Vector Machine (SVC)
4. Naive Bayes

**Technologies Used** - Python, Pandas (Data Manipulation), NumPy (Numerical Computations),Matplotlib & Seaborn (Visualization), Scikit-learn (Machine Learning Models)
