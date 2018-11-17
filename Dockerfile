FROM gw000/keras:2.1.4-py3

# install dependencies from debian packages
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    python-matplotlib \
    python-pillow \
    tesseract-ocr

# install dependencies from python packages
RUN pip3 --no-cache-dir install \
    pandas \
    scikit-learn \
    statsmodels \
    scipy \
    pytesseract \
    nltk

# install FLASK

# install your app
ADD images/ /tmp/test_script/
ADD extract_text.py /tmp/test_script/

RUN chmod +x /tmp/test_script/extract_text.py

# default command
CMD ["/tmp/test_script/extract_text.py"]
