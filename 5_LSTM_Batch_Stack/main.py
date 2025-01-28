"""
Before starting to the tutorial series:
In each tutorial, there's a main.py file that contains the main code snippet.
The main.py file is the one that you should run to execute the experiment.
In each main.py file, there is:
+ Model definition class (e.g. LSTMGenerator) that inherits from torch.nn.Module. There maybe additional model classes if needed.
  But at the end all will be wrapped in a final nn.Module class that will be used in the PyTorch Lightning module.
  Model class contains:
  + the neural network architecture
  + forwarding pass logic
  + loss function.
+ PyTorch Lightning module class (e.g. TextGenerationExperiment) that inherits from pytorch_lightning.LightningModule.
 + Experiment class contains LightningModule required methods plus:
   + model definition above
   + training_step method
   + validation_step method
   + configure_optimizers method
   + additional methods for testing performance per epoch, etc.
   + a forward method to generate output as a response to the end-user input.
The model and the experiment instances are glued together using the PlApp class.
"""

"""
5_Character_Generation_with_LSTM_Batch_Stacked
In this module, we stack the LSTM. there's a change in the NN architecture, we expect some litte drop in the loss value. Basically we only change the nn.LSTM constructor to have num_layers > 1.
Following classes are inside the external Python module base_text:
+ CharacterDataSet(data.IterableDataset): Parses and processes dataset.
+ CharacterDataModule(pl.LightningDataModule): Handles batching, data loading, utilizes CharacterDataSet.
"""

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import os
# add the project root folder to the python path so that we can import base modules
sys.path.append(os.getcwd()) 
import base
import base_textgen

# load configuration for the this module
params = base.PlApp.init_env(os.path.dirname(os.path.abspath(__file__)))
    
class LSTMGenerator(nn.Module):
    def __init__(self, input_output_size, embed_size, hidden_size, num_layers):
        """
        This is the simplest LSTM based generator model. 
        :param input_output_size: Number of unique characters in the dataset, i,e. vocab size.
        :param embed_size: Dimension of the embedding vector.
        :param hidden_size: Dimension of the hidden state.
        """
        super().__init__()
        # we have an embedding layer to convert character indexes to vectors, the weights are learned.
        self.embed = nn.Embedding(num_embeddings=input_output_size, embedding_dim=embed_size)
        # this is the LSTM that will process the input to produce the output for each time step 
        # the hidden and cell states are not part of the layer, so we need to declare,keep it and feed it in training loop.
        # module instance name is still rnn as LSTM is a type of RNN. Check batch_first description in forward method.
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # the output of the LSTM is used to predict the next character, so we have a fully connected layer from hidden state to the output size.
        self.fc = nn.Linear(hidden_size, input_output_size)

    def forward(self, characters, hc_states=None):
        """
        This method is triggered when the model(forward) is called like self.model(...) in the training_step method of the PyTorch Lightning module.
        Because __call__ method of the nn.Module class calls the forward method.
        We feed the single character index to the model and get the output logits for the next character.
        :param characters: List of character indices. [batch_size,time_steps]
        :param hc_states: Initial hidden and cell states pair, both [num_layers,batch_size,hidden_dim]. 
        """
        # embed layer accepts any shaped output. It just adds a new dimension holding the embedded vector.
        embedded = self.embed(characters)  # [batch_size,time_steps,embedding_dim]

        # LSTM layer accepts 3D input for batched data. When the batch_first is true [batch_size,time_steps,embedding_dim], else [time_steps,batch_size,embedding_dim] 
        # and optional initial hidden and cell states. It returns the output at each time step and final hidden and cell states pair.
        # hidden and cell states are also 3D tensors for batched data.
        # the first dimension is number of layers and it's specified in nn.LSTM constructor. So the hidden and cell states are [num_layers,batch_size,hidden_dim] tensors.
        # the as the output at each time step is the hidden state at that time step, output[i][-1] will have the same value as hc_states[0][0][i]
        output, hc_states = self.rnn(embedded, hc_states) 
        # output [batch_size,time_steps,hidden_dim] ([num_layers,batch_size,hidden_dim],[num_layers,batch_size,hidden_dim])

        # to speed up training, cross entropy expects unnormalized logits as input so we don't use softmax here.
        # we'll use the output to predict the next characters.
        fc_output = self.fc(output)  # [batch_size,time_steps,input_output_size]    
        return fc_output, hc_states

    # [batch_size*time_steps,input_output_size], [batch_size*time_steps]
    def loss_func(self, logits, characters):
        # cross entropy expects unnormalized logits as input and class index as target
        return F.cross_entropy(logits, characters)

class TextGenerationExperiment(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, prime_input, predict_len=100, temperature=0.25):
        """
        This method is triggered when the experminent(forward) is called like self(...) in the on_train_epoch_end method of the PyTorch Lightning module.
        Because __call__ method of LightningModule class calls the forward method similar to nn.Module.
        This method is not part of actual training process but generates text using the model trained so far. This is used to track the progress of the model 
        from the generated text in the tensorboard.
        :param prime_input: Initial text to start the generation.
        :param predict_len: Length of the generated text.
        :param temperature: Determines the sharpness of the probability distribution.
        """
        # convert the prime input text to tensor
        prime_tensor = base_textgen.CharacterDataSet.char_to_tensor(prime_input).to(self.device) # [prime_input_len]

         # generations will keep the generated characters as integer indexes. But of course the first part is the prime input.
        generations = [t.item() for t in prime_tensor]

        # initial hidden and cell states are None, so the model will use zeros.
        hc_states = None
        output = None

        # build the hidden state for the prime input and get the first output
        output, hc_states = self.model(prime_tensor, hc_states) 
        # output [prime_tensor_len, input_output_size], hc_states ([num_layers,hidden_dim],[num_layers,hidden_dim])

        # as the model output is from a fully connected layer, it is not normalized. To be able to use it as a probability distribution,
        # we'll use torch.multinomial which handles normalization (summing to 1) internally. but first the values should be non-negative.
        # So we'll use exp to make them positive which also keeps the order.
        for _ in range(predict_len):
            # the output first comes from forwarding of prime_tensor, so it'll have multiple time steps.
            # Here we get the output of last time step. Also when we get the output again in the loop, even if we pass one character, there'll be a time step dimension.
            output = output[-1]  # [input_output_size]
            # Additionally we'll change how sharp the probability distribution be for the maximum one. This is called temperature.
            # Because when the outputs divided by the floating number in the .0..1. range,
            # if the output value is negative, it'll be more negative, if it is positive, it'll be more positive.
            # The exponentiation also amplifies the distance between the values.
            output_dist = output.div(temperature).exp()  # [input_output_size] # e^{logits / T}
            
            # sample from the distribution. multinomial keeps dimension so it's OK to pass it again to the model as it is.
            t = torch.multinomial(output_dist, 1)  # 1: sample only one, t [1]
            generations.append(t.item())

            # feed the generated character to the model to get the next character
            output, hc_states = self.model(t, hc_states)  # output [input_output_size], hc_states([1,hidden_dim],[1,hidden_dim])
        gen_text = ''.join([base_textgen.CharacterDataSet.vocab[t] for t in generations])
        return gen_text

    def training_step(self, data, data_idx):
        """
        This is the most important method in the PyTorch Lightning module. It is called by trainer.fit, when data module 
        yields a (batch) of data. In this case, as we don't use batching, it is called for each data pack.
        As per params.yml this method will be called iters_per_epoch per epoch.
        :param data: A list containing 0:input and target tensors. As per params.yml this has a size of portion_size.
        :param data_idx: Index of the data.
        """
        loss = 0
        # as it can be seen in the __iter__ method of the CharacterDataSet, each data pack contains a fixed size but random portion of the whole text as input and one character shifted version of it as target. So the inputs and targets are 1d tensor of character indexes.
        # in sequence processing jargon, each character (in other words each element of the tensor) is a time step.
        # so instead of portion_size, we'll use time_steps to refer the length of the sequence.
        input, target = data # inputs [batch_size,time_steps], targets [batch_size, time_steps]
        """
        For example:
        input[i]:   tensor([13, 14, 16, 18, 28, 29, 18, 27, 14, 11, 18, 21, 14, 12, 14, 20, 36, 16,
                    30, 12, 14, 36, 28, 10, 17, 18, 25, 36, 24, 21, 13, 30, 16, 30, 22, 30,
                    35, 36, 35, 10, 22, 10, 23, 36, 11, 30, 36, 17, 10, 21], device='cuda:0')
        target[i]:  tensor([14, 16, 18, 28, 29, 18, 27, 14, 11, 18, 21, 14, 12, 14, 20, 36, 16, 30,
                    12, 14, 36, 28, 10, 17, 18, 25, 36, 24, 21, 13, 30, 16, 30, 22, 30, 35,
                    36, 35, 10, 22, 10, 23, 36, 11, 30, 36, 17, 10, 21, 20], device='cuda:0')
        """
        hc_states = None

        # as we use LSTM instead of cells, we don't need to loop through the time steps. We can process all at once.
        output, hc_states = self.model(input, hc_states)
        # output [batch_size,time_steps,input_output_size], hc_states ([1,batch_size,hidden_dim],[1,batch_size,hidden_dim])

        # Since loss function expects a maximum of 2D input in the output and 1D or 2D for the ground truth,
        # we'll need to flatten the output and target tensors.

        output = output.flatten(0, 1)  # [batch_size*time_steps, input_output_size]
        target = target.flatten()  # [batch_size*time_steps]

        loss = self.model.loss_func(output, target)

        # here we log training loss to the logging system so that we can see it in the tensorboard.
        # also by setting prog_bar=True, it will be shown in the progress bar in the console.

        # regarding the loss value logged in varius ways:
        # + on_step=False, on_epoch=True
        #   + progress bar: we'll see a value after first epoch ended and updated after each epoch.
        #   + tensorboard: we'll see the same value shown in the progress bar in the tensorboard.
        #     We'll see the logs at every step that matches to the batch size (if batch size is 32, we'll see at 32, 64, 96, etc. in the graph).
        #     Because, by default, logger system logs at every epoch end.
        #   + value:  The value won't be the loss of the last step of the epoch, but some average of the all self.log()'ed values in that epoch.
        # + on_step=True, on_epoch=False
        #   + progress bar: we'll see a value at each step.
        #   + tensorboard: as log_every_n_steps parameter of the trainer is 50 by default, we'll see logged values at steps 0, 49, 99, 149, etc in tensorboard.
        #   + value:  The value will be the exact loss of the that step.      
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """
        This method is called at the end of each epoch. We use it to generate text using the model and log it to the tensorboard.
        However when called model is in training mode, so we need to switch it to evaluation mode to generate text.
        Also we need to disable gradient calculation to save memory and computation time.
        """
        self.model.eval()
        with torch.no_grad():
            gen_text = self(**params["run"])
        
        # restore the model to the training mode
        self.model.train()

        # log the generated text to the tensorboard
        self.logger.experiment.add_text(tag="Generated", text_string=gen_text, global_step=self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)



# init data module using data section of the configuration
p = params['data']
dataset = base_textgen.CharacterDataSet(**p)
data_module = base_textgen.BatchingDataModule(dataset, p['batch_size'])

# init model using model section of the configuration
p = params['model']
model = LSTMGenerator(input_output_size=len(dataset.vocab), **p)

# init experiment
experiment = TextGenerationExperiment(model=model,**params['experiment'])

# init PyTorch Lightning application by referencing data module, model instances and the experiment class
pl_app = base.PlApp(data_module=data_module, model=model, experiment=experiment, params=params)

# start training
pl_app.train()
