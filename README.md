# Fake News Classification
This project focuses on detecting fake news using various ML models. We compare an ensemble model, a fine-tuned BERT, and FastText to find the most effective approach.
<hr>

### Data
The dataset fake_news_full_data.csv includes headlines, article texts, publication dates, and labels (0 — real, 1 — fake). It is balanced: ~52.5% fake news.

### Approach
Preprocessing: text cleaning, tokenization, stopword removal, lemmatization.
<br>EDA: fake news often includes words like trump, people; real news — said. Very short texts (<100) or long ones (1500–2200 chars) tend to be fake.

### Models
<table>
  <thead>
    <tr>
      <th style="text-align:center">Model</th>
      <th style="text-align:center">Train F1 Score</th>
      <th style="text-align:center">Validation F1 Score</th>
      <th style="text-align:center">Train Accuracy</th>
      <th style="text-align:center">Validation Accuracy</th>
      <th style="text-align:center">Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">Ensemble (LR + DT)</td>
      <td style="text-align:center">98.79%</td>
      <td style="text-align:center">98.26%</td>
      <td style="text-align:center">98.79%</td>
      <td style="text-align:center">98.26%</td>
      <td style="text-align:center">TF-IDF</td>
    </tr>
    <tr>
      <td style="text-align:center">Fine-tuned BERT</td>
      <td style="text-align:center">99.36%</td>
      <td style="text-align:center">99.36%</td>
      <td style="text-align:center">99.36%</td>
      <td style="text-align:center">99.36%</td>
      <td style="text-align:center">BERT base uncased</td>
    </tr>
    <tr>
      <td style="text-align:center">FastText</td>
      <td style="text-align:center">99.99%</td>
      <td style="text-align:center">99.87%</td>
      <td style="text-align:center">99.99%</td>
      <td style="text-align:center">99.87%</td>
      <td style="text-align:center"><code>__label__</code> format</td>
    </tr>
  </tbody>
</table>
<hr>

### Conclusion
FastText is the most accurate and efficient. BERT offers high precision but requires more resources. The ensemble is a solid, interpretable baseline.

