import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Load the pre-trained model and tokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model= torch.load(r'D:\STD_DESK\STDs\Fake_News_Detection\15-group-PROJECT-FOLDER\Research Paper\CODES\WELFake\WELFake_trained_model.pth', map_location=torch.device('cpu'), weights_only=False)

# Replace with your model path
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class FakeNewsDataset:  
    def __init__(self, statement, tokenizer, max_len=512):
        self.statement = statement
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        word_pieces = ['[CLS]']
        statement = self.tokenizer.tokenize(self.statement)
        word_pieces += statement + ['[SEP]']
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        if len(tokens_tensor) < self.max_len:
            tokens_tensor = torch.cat([tokens_tensor, torch.zeros(self.max_len - len(tokens_tensor), dtype=torch.long)])
        else:
            tokens_tensor = tokens_tensor[:self.max_len]

        segments_tensor = torch.tensor([0] * len(word_pieces), dtype=torch.long)
        if len(segments_tensor) < self.max_len:
            segments_tensor = torch.cat([segments_tensor, torch.zeros(self.max_len - len(segments_tensor), dtype=torch.long)])
        else:
            segments_tensor = segments_tensor[:self.max_len]

        return tokens_tensor, segments_tensor

    def __len__(self):
        return 1

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return tokens_tensors, segments_tensors, masks_tensors

def predict_fake_news(file=None, text=None):
    if file:
        # Read the file content
        try:
            with open(file.name, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}", 0.0
    
    if not text:
        return "Error: No text provided", 0.0

    # Process the text as before
    dataset = FakeNewsDataset(text, tokenizer)
    loader = DataLoader(dataset, batch_size=1, collate_fn=create_mini_batch)
    with torch.no_grad():
        model.eval()
        for data in loader:
            tokens_tensors, segments_tensors, masks_tensors = [t.to(device) for t in data]
            outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidence_true = probabilities[0][1].item()
            confidence_fake = probabilities[0][0].item()

            prediction = "Real" if confidence_true > confidence_fake else "Fake"
            confidence_score = max(confidence_true, confidence_fake)

            return prediction, confidence_score
# def predict_fake_news(text):
#     dataset = FakeNewsDataset(text, tokenizer)
#     loader = DataLoader(dataset, batch_size=1, collate_fn=create_mini_batch)
#     with torch.no_grad():
#         model.eval()
#         for data in loader:
#             if next(model.parameters()).is_cuda:
#                 data = [t.to(device) for t in data]
#             # tokens_tensors, segments_tensors, masks_tensors = data
#             tokens_tensors, segments_tensors, masks_tensors = [t.to(device) for t in data]
#             outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
#             # logits = outputs[0]  
#             logits = outputs.logits  # Use 'outputs.logits' if outputs is a dictionary
#             probabilities = torch.softmax(logits, dim=1) 
#             print(probabilities.shape)
#             confidence_true = probabilities[0][1].item()  
#             confidence_fake = probabilities[0][0].item()  

#             prediction = "Real" if confidence_true > confidence_fake else "Fake"
#             confidence_score = max(confidence_true, confidence_fake)

#             return prediction, confidence_score


iface = gr.Interface(
    fn=predict_fake_news,
    inputs=[
        gr.File(label="Upload a .txt or .csv file"),  # Allow file uploads
        gr.Textbox(lines=5, placeholder="Or paste your article or headline here..."),  # Optional text input
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Number(label="Confidence Score of Prediction"),
    ],
    title="Fake News Detector",
    description="Upload a file or paste text to detect fake news. Powered by BERT",
    allow_flagging="never",
)

iface.launch()