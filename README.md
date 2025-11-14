# EEG Motor Imagery Classification for BCI Systems

This project implements a Brain-Computer Interface (BCI) system for classifying EEG signals associated with imagined hand movements (Motor Imagery) using classical machine learning techniques. It is based on the thesis developed at Sapienza Università di Roma.

## 🎯 Objective

Develop a robust and interpretable pipeline for classifying EEG signals related to motor imagery (left vs. right hand) using preprocessing, feature extraction, and machine learning classifiers.

## 🧠 Background

Motor Imagery EEG classification is a key challenge in non-invasive BCI systems. EEG signals are noisy and non-stationary, requiring careful preprocessing and feature engineering. This project focuses on:

- Preprocessing EEG signals from the BCI Competition IV - Dataset 2b
- Extracting features using Common Spatial Pattern (CSP), log-variance, and PSD
- Classifying signals using Support Vector Machine (SVM) and Linear Discriminant Analysis (LDA)
- Evaluating performance using 10-fold cross-validation

## 🛠 Technologies Used

- Python
- MNE-Python
- NumPy, SciPy
- scikit-learn
- Matplotlib, Seaborn

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/lucacataldo1106/motor-imagery-eeg-classification.git
cd motor-imagery-eeg-classification
pip install -r requirements.txt
```
## 📁 Project Structure
```
motor-imagery-eeg-classification/
├── data/                  # EEG dataset files (BCI Competition IV - 2b)
├── preprocessing/         # Signal filtering and artifact removal
├── features/              # Feature extraction scripts (CSP, PSD, log-var)
├── classification/        # SVM and LDA classifiers
├── results/               # Output metrics and plots
├── README.md
└── requirements.txt
```
## 🚀 Usage

To run the full classification pipeline:

```bash
python main.py
```
## 📊 Results

- **Dataset**: BCI Competition IV - Dataset 2b  
- **Channels Used**: C3, Cz, C4  
- **Sampling Rate**: 250 Hz  
- **Classification Accuracy**: 70.9% (subject-dependent, 10-fold cross-validation)  
- **Classifiers Used**: Support Vector Machine (SVM), Linear Discriminant Analysis (LDA)

## 🔬 Future Work

- Integrate filter bank CSP and feature selection techniques (e.g., PSO, NCA)
- Explore wavelet-based multi-scale feature extraction
- Compare performance with deep learning models (CNN, RNN)
- Extend pipeline to cross-subject validation and real-time classification

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

- Dataset provided by BCI Competition IV  
- Thesis developed at Sapienza Università di Roma  
- Supervised by Prof. Danilo Avola  
- Authored by Luca Cataldo (Matricola 1911906)

## 📬 Contact

For questions, feedback, or collaboration opportunities:  
**Author**: Luca Cataldo
**Email**: luca.cataldo1106@gmail.com  
**GitHub**: [@lucacataldo1106](https://github.com/lucacataldo1106)
**LinkedIn**: https://www.linkedin.com/in/luca-cataldo1106/
