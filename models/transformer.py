import torch
import torch.nn as nn
import numpy as np

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, num_heads, dim_feedforward, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see nn.TransformerEncoderLayer
        :param num_heads: see nn.TransformerEncoderLayer
        :param num_heads: see nn.TransformerEncoderLayer
        :param num_classes: number of classes predicted at the output layer; should be 2 [ace, not ace]
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        
        # check that num_postions is 20
        # if num_positions != 20:
        #     raise Exception("num_positions must be 20")
        
        # # check that num_classes is 3
        # if num_classes != 3:
        #     raise Exception("num_classes must be 3")

        # initialize embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        
        # initialize positional encoding
        self.positional_encoding = PositionalEncoding(d_model, num_positions)

        # initialize encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # initialize the output layer
        self.output_layer = nn.Linear(d_model, num_classes)

        # initialize log softmax
        self.log_softmax = nn.LogSoftmax(dim=-1)


    def forward(self, indices, mask=None):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """

        # get log probabilities
        log_probs = self.log_softmax(self.output_layer(self.encoder(self.positional_encoding(self.embedding_layer(indices)), mask=mask)))

        output = log_probs.permute(0, 2, 1)  # Change shape to (batch_size, 2, seq_len)


        # get attention maps
        # attention_maps = [layer.attention_map for layer in self.transformer_layers]

        # return log probabilities
        return output
        

# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)