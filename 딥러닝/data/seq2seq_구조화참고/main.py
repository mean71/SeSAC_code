import torch 
import torch.nn as nn

from data_handler import create_integer_sequence_dataset, create_lang_pair, Vocabulary
from seq2seq import Encoder, Decoder, Seq2Seq
from trainer import Trainer 

def train_integer_sequence(
        # model-related options (hyperparamter)
        hidden_size = 128, 
        embedding_dim = 4, 
        encoder_type = 'lstm', 
        decoder_type = 'lstm', 
        # data-related options 
        num_samples = 1000, 
        max_input_len = 10, 
        max_target_len = 15, 
        vocab_size = 10, 
        batch_size = 1000, 
        # training options 
        num_epochs = 100,  
        learning_rate = 0.003, 
        optimizer = torch.optim.Adam, 
        criterion = nn.CrossEntropyLoss, ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create dataset
    (train_loader, valid_loader, test_loader), vocab = create_integer_sequence_dataset(
        num_samples = num_samples, 
        max_input_len = max_input_len, 
        max_target_len = max_target_len, 
        vocab_size = vocab_size, 
        batch_size = batch_size, 
    )

    # 2. Build the model 
    encoder = Encoder(
        vocab_size = vocab.n_words,
        embedding_dim = embedding_dim,
        hidden_size = hidden_size,
        model_type = encoder_type,
    ).to(device)

    decoder = Decoder(
        vocab_size = vocab.n_words,
        embedding_dim = embedding_dim,
        hidden_size = hidden_size,
        model_type = decoder_type, 
    ).to(device)

    model = Seq2Seq(encoder, decoder, device).to(device)

    # 3. Train the model 
    optimizer = optimizer(model.parameters(), lr = learning_rate)
    criterion = criterion(ignore_index = vocab.pad_idx)

    trainer = Trainer(
        model = model,
        device = device,
        criterion = criterion,
        optimizer = optimizer,
        save_dir = 'models',
    )

    # Train the model
    trainer.train(
        train_loader = train_loader,
        valid_loader = valid_loader,
        num_epochs = num_epochs,
        print_every = 1,
        evaluate_every = 1,
    )

    # 4. Evaluate the model 
    trainer.load_best_model()

    valid_loss, valid_acc = trainer.evaluate(valid_loader = test_loader)
    train_loss, train_acc = trainer.evaluate(valid_loader = train_loader)
    print(f'Best model validation loss: {valid_loss:.4f}, validation acc: {valid_acc:.4f}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
    
    example_input, example_target = next(iter(valid_loader))
    example_input = example_input[0].tolist()
    example_target = example_target[0].tolist()
    print('Input sequence:', [vocab.index2word[idx] for idx in example_input if idx not in range(len(Vocabulary.SPECIAL_TOKENS))])
    print('Target sequence:', [vocab.index2word[idx] for idx in example_target if idx not in range(len(Vocabulary.SPECIAL_TOKENS))])

    translated_sentence = trainer.inference(example_input, vocab)
    print('Predicted sequence:', translated_sentence)

    return trainer 

def train_machine_translator(
        # model-related options (hyperparamter)
        hidden_size = 128, 
        embedding_dim = 64, 
        encoder_type = 'lstm', 
        decoder_type = 'lstm', 
        # training options 
        num_epochs = 100,  
        learning_rate = 0.005   , 
        optimizer = torch.optim.Adam, 
        criterion = nn.CrossEntropyLoss, ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Create dataset
    (train_loader, valid_loader, test_loader), source_vocab, target_vocab \
            = create_lang_pair('kor.txt')
    
    # 2. Build the model 
    encoder = Encoder(
        vocab_size = source_vocab.n_words, 
        embedding_dim = embedding_dim, 
        hidden_size = hidden_size, 
        model_type = encoder_type, 
    ).to(device)

    decoder = Decoder(
        vocab_size = target_vocab.n_words, 
        embedding_dim = embedding_dim, 
        hidden_size = hidden_size, 
        model_type = decoder_type, 
    ).to(device)

    model = Seq2Seq(encoder, decoder, device).to(device) 

    # 3. Train the model 
    optimizer = optimizer(model.parameters(), lr = learning_rate)
    criterion = criterion(ignore_index = target_vocab.pad_idx)

    trainer = Trainer(
        model = model,
        device = device,
        criterion = criterion,
        optimizer = optimizer,
        save_dir = 'machine_translation',
    )

    for epoch in range(num_epochs//10):
        trainer.train(
            train_loader = train_loader,
            valid_loader = valid_loader,
            num_epochs = 10,
            print_every = 1,
            evaluate_every = 1,
            evaluate_metric = 'BLEU'
        )

        # 4. Evaluate the model 
        valid_loss, valid_acc = trainer.evaluate(valid_loader = test_loader)
        train_loss, train_acc = trainer.evaluate(valid_loader = train_loader)
        print(f'Best model validation loss: {valid_loss:.4f}, validation acc: {valid_acc:.4f}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        
        example_input, example_target = next(iter(train_loader))
        example_input = example_input[0].tolist()
        example_target = example_target[0].tolist()
        # print('Input sequence:', [source_vocab.index2word[idx] for idx in example_input if idx not in range(len(Vocabulary.SPECIAL_TOKENS))])
        # print('Target sequence:', [target_vocab.index2word[idx] for idx in example_target if idx not in range(len(Vocabulary.SPECIAL_TOKENS))])
        print('Input sequence:', [source_vocab.index2word[idx] for idx in example_input])
        print('Target sequence:', [target_vocab.index2word[idx] for idx in example_target])

        translated_sentence = trainer.inference(example_input, target_vocab)
        print('Predicted sequence:', translated_sentence)

    return trainer 


if __name__ == '__main__':
    train_integer_sequence()
    # train_machine_translator()