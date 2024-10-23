def log_results(model_name, accuracy, precision, recall, training_time):
    """
    Print and log the results of the model evaluation.
    """
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Training time: {training_time:.4f} seconds\n")
