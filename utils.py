import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from matplotlib.backends.backend_pdf import PdfPages

def send_email_notification(content="test"):
    email = "benreyhani15@gmail.com"
    msg = MIMEMultipart()
    msg["From"] = email
    msg["To"] = email
    msg["Subject"] = "Done - Linear SVM with l1"
    msg.attach(MIMEText(content))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email, "Rapsrule23!")
    text = msg.as_string()
    server.sendmail(email, email, text)
    server.quit()

def write_plots_to_pdf(path, plots_array, notes_array, title):
    print()
    #with PdfPages
# TODO: Write binary importance scatter plots to pdf; a plot for each page ; a page for each component; place in dataset folder 
    # under new directory ; also, attached the notes for important frequencies and placce the min and max frequencies somewhere