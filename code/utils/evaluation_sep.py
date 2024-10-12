
from omegaconf import OmegaConf
import torch
from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image
import pandas as pd
import io
import IPython.display as display
from tqdm import tqdm
# Assuming your JSONParseEvaluator class is implemented as described above

def post_process_prediction(generated_text: str):
    """
    Post-process the model output to clean malformed JSON-like predictions.
    Here you would need to handle closing of tags, malformed sequences, and proper token mapping.
    """
    # Replace special characters, normalize spaces, etc.
    # For example: You might replace some control characters, fix tag issues, etc.
    clean_text = generated_text.replace('<0x0A>', ' ').replace('\n', ' ').strip()
    
    # Further processing might be needed depending on the structure of the output
    return clean_text

def run_evaluation(
        cfg: OmegaConf, 
        model, 
        valid_dl, 
        tokenizer, 
        token_map):

    # Config for text generation
    conf_g = {
        "max_new_tokens": cfg.model.max_length_generation,  # 256,
        "do_sample": False,
        "top_k": 1,
        "use_cache": True,
    }

    generation_config = torch.nn.GenerationConfig(**conf_g)

    # Set model to evaluation mode
    model.eval()

    all_ids = []
    all_texts = []
    label_dict = []
    progress_bar = tqdm(range(len(valid_dl)), desc='Running evaluation...')

    for batch in valid_dl:
        with torch.no_grad():
            batch_ids = batch["id"]
            
            generated_ids = model.backbone.generate(
                flattened_patches=batch['flattened_patches'],
                attention_mask=batch['attention_mask'],
                generation_config=generation_config,
            )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_ids.extend(batch_ids)
            all_texts.extend(generated_texts)
            label_dict.extend(batch['texts'])
        
        progress_bar.update(1)
    progress_bar.close()

    # Post-process ground truth and predictions
    label_dicts = [
        post_process_prediction(label_str) for label_str in label_dict
    ]

    preds_dict = []
    for this_id, this_text in zip(all_ids, all_texts):
        pred_dictionary = post_process_prediction(this_text)
        preds_dict.append((this_id, pred_dictionary))

    eval_JSON = JSONParseEvaluator()

    # Calculate F1 score
    f1_score = eval_JSON.cal_f1(
        preds=preds_dict,
        answers=label_dicts
    )

    # Calculate accuracy
    accuracy = sum([
        eval_JSON.cal_acc(pred, label)
        for pred, label in zip(preds_dict, label_dicts)
    ]) / len(preds_dict)

    return {
        'f1_score': f1_score,
        'accuracy': accuracy
    }


# Main function to load the model, process the parquet file, and run the evaluation
if __name__ == "__main__":
    # Load the configuration file
    cfg = OmegaConf.load("/content/conf_r_final.yaml")

    # Set the path to the parquet file
    parquet_file = cfg.custom.valid_parquet_path

    # Load model and processor from the configuration
    model_dir = cfg.output.model_dir
    model = Pix2StructForConditionalGeneration.from_pretrained(model_dir)
    processor = Pix2StructProcessor.from_pretrained(model_dir)

    # Load the validation dataset from the parquet file
    valid_df = pd.read_parquet(parquet_file)

    # Create a DataLoader-like structure
    # valid_dl = [{'id': row['id'], 'flattened_patches': processor(row['image']), 'texts': row['text']} for _, row in valid_df.iterrows()]

    # Run the evaluation
    # results = run_evaluation(cfg, model, valid_dl, processor, token_map={})
    
    # Print results
    # print(f"F1 Score: {results['f1_score']}")
    # print(f"Accuracy: {results['accuracy']}")

    image_bytes = valid_df['image'][0]['bytes']
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((512, 512))  # Ensure image is resized to 512x512
    display.display(image)  # Display the image inside the notebook

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Make sure the model is on GPU

    inputs = processor(
        images=image,
        return_tensors="pt",  # Ensure the output is a PyTorch tensor
        max_patches=1024,  # Configurable max_patches
        add_special_tokens=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    generation_config = {
        "max_length": 1024,  # Maximum length of generated sequence
        "min_length": 512,   # Minimum length of generated sequence
        "no_repeat_ngram_size": 3,  # Avoid repetition of 3-grams
        "early_stopping": False,    # Continue generating until max_length
        "num_beams": 5,  # Optional: Beam search to improve generation quality
    }

    with torch.no_grad():
        # Generation method for the single image
        outputs = model.generate(
            inputs['flattened_patches'],  # Pass the processed image patches
            attention_mask=inputs['attention_mask'],
            max_length=generation_config["max_length"],  # Max length from config
            min_length=generation_config["min_length"],  # Min length from config
            no_repeat_ngram_size=generation_config["no_repeat_ngram_size"],  # Avoid repetition
            early_stopping=generation_config["early_stopping"],  # Control stopping criteria
            num_beams=generation_config["num_beams"],  # Beam search for better quality
        )

# Decode the generated output using the processor
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
    print("Generated Output:", generated_text)
