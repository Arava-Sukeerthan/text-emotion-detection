from transformers import pipeline

print("Loading Hugging Face model...")

classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions"
)

text = input("Enter your sentence: ")

result = classifier(text)

print("\nResult:")
print(result)