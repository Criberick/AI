#Wymaga instalacji pandas, PyQt, scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from PyQt5.QtWidgets import QComboBox, QMainWindow, QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QDialog, QLabel, QMessageBox, QFileDialog, QLabel, QMessageBox
from PyQt5.QtGui import QIcon, QIntValidator, QRegExpValidator, QDoubleValidator
from PyQt5.QtCore import QRegExp, pyqtSlot

import numpy as np
import sys


class MultiColumnLabelEncoder:
    def __init__(self, columns):
        self.columns = columns  # array of column names to encode
        # 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        self.gender_enc = LabelEncoder()
        self.married_enc = LabelEncoder()
        self.dependents_enc = LabelEncoder()
        self.education_enc = LabelEncoder()
        self.self_employed_enc = LabelEncoder()
        self.property_enc = LabelEncoder()
        self.name_map = {'Gender': self.gender_enc, 'Married': self.married_enc, 'Dependents': self.dependents_enc,
                         'Education': self.education_enc, 'Self_Employed': self.self_employed_enc, 'Property_Area': self.property_enc}

    def fit(self, X):
        if self.columns is not None:
            for col in self.columns:
                self.name_map[col].fit(X[col])

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = self.name_map[col].transform(output[col])
        return output

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


df = pd.read_csv("loan-train.csv")
df = df.drop(columns=["Loan_ID"])
df = df[df[['Gender', 'Married', 'Dependents', 'Self_Employed',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History']].notnull().all(1)]

X = df[df.columns[:-1]]
enc = MultiColumnLabelEncoder(columns=[
                              'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
X = enc.fit_transform(X)
y = df[df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix: {cm}")
print(f"Cross validation: {classifier.score(X_test, y_test)}")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        lbl_gender = QLabel('Gender')
        lbl_married = QLabel('Married')
        lbl_dependents = QLabel('Dependents')
        lbl_education = QLabel('Education')
        lbl_self_employed = QLabel('Self employed')
        lbl_applicant_income = QLabel('Applicant income')
        lbl_coapplicant_income = QLabel('Coapplicant income')
        lbl_loan_amount = QLabel('Loan amount')
        lbl_loan_term = QLabel('Loan amount term')
        lbl_credit_history = QLabel('Credit history')
        lbl_area = QLabel('Property area')


        self.cb_gender = QComboBox()
        self.cb_gender.addItems(['Male', 'Female'])

        self.cb_married = QComboBox()
        self.cb_married.addItems(['Yes', 'No'])

        self.cb_dependents = QComboBox()
        self.cb_dependents.addItems(['0', '1', '2', '3+'])

        self.cb_education = QComboBox()
        self.cb_education.addItems(['Graduate', 'Not Graduate'])

        self.cb_self_employed = QComboBox()
        self.cb_self_employed.addItems(['Yes', 'No'])

        self.le_applicant_income = QLineEdit()
        self.le_applicant_income.setValidator(QIntValidator())
        self.le_applicant_income.setPlaceholderText("4583")

        self.le_coapplicant_income = QLineEdit()
        self.le_coapplicant_income.setValidator(QDoubleValidator())
        self.le_coapplicant_income.setPlaceholderText("1508.0")

        self.le_loan_amount = QLineEdit()
        self.le_loan_amount.setValidator(QDoubleValidator())
        self.le_loan_amount.setPlaceholderText("128.0")

        self.exp_loan_term = QRegExp(
            "^(360.|120.|180.|60.|300.|480.|240.|36.|84.)$")
        self.le_loan_term = QLineEdit()
        self.le_loan_term.setValidator(QRegExpValidator(self.exp_loan_term))
        self.le_loan_term.setPlaceholderText("360.")

        self.exp_credit_history = QRegExp("^(1.|0.)$")
        self.le_credit_history = QLineEdit()
        self.le_credit_history.setValidator(
            QRegExpValidator(self.exp_credit_history))
        self.le_credit_history.setPlaceholderText("1.")

        self.cb_area = QComboBox()
        self.cb_area.addItems(['Rural', 'Urban', 'Semiurban'])

        btn_calculate = QPushButton('Calculate')
        btn_calculate.clicked.connect(self.on_click)
        
        btn_export_to_csv = QPushButton('Export CSV')
        btn_export_to_csv.clicked.connect(self.export_csv)


        layout = QVBoxLayout()
        layout.addWidget(lbl_gender)
        layout.addWidget(self.cb_gender)
        layout.addWidget(lbl_married)
        layout.addWidget(self.cb_married)
        layout.addWidget(lbl_dependents)
        layout.addWidget(self.cb_dependents)
        layout.addWidget(lbl_education)
        layout.addWidget(self.cb_education)
        layout.addWidget(lbl_self_employed)
        layout.addWidget(self.cb_self_employed)
        layout.addWidget(lbl_applicant_income)
        layout.addWidget(self.le_applicant_income)
        layout.addWidget(lbl_coapplicant_income)
        layout.addWidget(self.le_coapplicant_income)
        layout.addWidget(lbl_loan_amount)
        layout.addWidget(self.le_loan_amount)
        layout.addWidget(lbl_loan_term)
        layout.addWidget(self.le_loan_term)
        layout.addWidget(lbl_credit_history)
        layout.addWidget(self.le_credit_history)
        layout.addWidget(lbl_area)
        layout.addWidget(self.cb_area)
        layout.addWidget(btn_calculate)
        layout.addWidget(btn_export_to_csv)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    @pyqtSlot()
    def on_click(self):
        if not self.exp_loan_term.exactMatch(self.le_loan_term.text()) or not self.exp_credit_history.exactMatch(self.le_credit_history.text()):
            print("Value does not match the regex!")
            return

        frame = pd.DataFrame.from_dict({
            # str
            "Gender":                  [str(self.cb_gender.currentText())],
            'Married':                  [str(self.cb_married.currentText())],
            'Dependents':               [str(self.cb_dependents.currentText())],
            'Education':      [str(self.cb_education.currentText())],
            'Self_Employed':        [str(self.cb_self_employed.currentText())],
            # numpy.int64
            'ApplicantIncome':      np.fromstring(self.le_applicant_income.text(), dtype=np.int64, sep=' '),
            # numpy.float64
            'CoapplicantIncome':    np.fromstring(self.le_coapplicant_income.text(), dtype=np.float64, sep=' '),
            # numpy.float64
            'LoanAmount':     np.fromstring(self.le_coapplicant_income.text(), dtype=np.float64, sep=' '),
            # numpy.float64
            'Loan_Amount_Term':  np.fromstring(self.le_loan_amount.text(), dtype=np.float64, sep=' '),
            # numpy.float64
            'Credit_History':    np.fromstring(self.le_credit_history.text(), dtype=np.float64, sep=' '),
            'Property_Area': [str(self.cb_area.currentText())]
        })
        transformed_frame = enc.transform(frame)
        predicted_value = classifier.predict(transformed_frame)
        msg_box = QMessageBox(self)
        if predicted_value[0] == 'Y':
            msg_box.setText("Approved the loan application!")
        else:
            msg_box.setText("Rejected the loan application!")
        msg_box.exec()

    def export_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if fileName:
            frame = pd.DataFrame.from_dict({
                # str
                "Gender": [str(self.cb_gender.currentText())],
                'Married': [str(self.cb_married.currentText())],
                'Dependents': [str(self.cb_dependents.currentText())],
                'Education': [str(self.cb_education.currentText())],
                'Self_Employed': [str(self.cb_self_employed.currentText())],
                # numpy.int64
                'ApplicantIncome': np.fromstring(self.le_applicant_income.text(), dtype=np.int64, sep=' '),
                # numpy.float64
                'CoapplicantIncome': np.fromstring(self.le_coapplicant_income.text(), dtype=np.float64, sep=' '),
                # numpy.float64
                'LoanAmount': np.fromstring(self.le_coapplicant_income.text(), dtype=np.float64, sep=' '),
                # numpy.float64
                'Loan_Amount_Term': np.fromstring(self.le_loan_amount.text(), dtype=np.float64, sep=' '),
                # numpy.float64
                'Credit_History': np.fromstring(self.le_credit_history.text(), dtype=np.float64, sep=' '),
                'Property_Area': [str(self.cb_area.currentText())]
            })
            transformed_frame = enc.transform(frame)
            predicted_value = classifier.predict(transformed_frame)
            frame['Loan_Status'] = predicted_value[0]
            frame['Probability'] = classifier.predict_proba(transformed_frame)[:, 1]

            frame.to_csv(fileName, index=False)
            QMessageBox.information(self, "Export CSV", "Data exported successfully!")

        
app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec_()