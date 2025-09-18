from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
import logging
import math
from torch.distributions import Categorical
from typing import Tuple, Union, Optional, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)
RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
RnnStateStorage = Tuple[torch.Tensor, ...]


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.

    # Parameters

    tensor : `torch.FloatTensor`, required.
        A batch first Pytorch tensor.
    sequence_lengths : `torch.LongTensor`, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    # Returns

    sorted_tensor : `torch.FloatTensor`
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : `torch.LongTensor`
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : `torch.LongTensor`
        Indices into the sorted_tensor such that
        `sorted_tensor.index_select(0, restoration_indices) == original_tensor`
    permutation_index : `torch.LongTensor`
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

def get_lengths_from_binary_sequence_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    # Parameters

    mask : `torch.BoolTensor`, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    # Returns

    `torch.LongTensor`
        A torch.LongTensor of shape (batch_size,) representing the lengths
        of the sequences in the batch.
    """
    return mask.sum(-1)

class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return type(self), (self.message,)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


class _EncoderBase(torch.nn.Module):

    """
    This abstract class serves as a base for the 3 `Encoder` abstractions in AllenNLP.
    - [`Seq2SeqEncoders`](./seq2seq_encoders/seq2seq_encoder.md)
    - [`Seq2VecEncoders`](./seq2vec_encoders/seq2vec_encoder.md)

    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """

    def __init__(self, stateful: bool = False) -> None:
        super().__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(
        self,
        module: Callable[
            [PackedSequence, Optional[RnnState]],
            Tuple[Union[PackedSequence, torch.Tensor], RnnState],
        ],
        inputs: torch.Tensor,
        mask: torch.BoolTensor,
        hidden_state: Optional[RnnState] = None,
    ):
        """
        This function exists because Pytorch RNNs require that their inputs be sorted
        before being passed as input. As all of our Seq2xxxEncoders use this functionality,
        it is provided in a base class. This method can be called on any module which
        takes as input a `PackedSequence` and some `hidden_state`, which can either be a
        tuple of tensors or a tensor.

        As all of our Seq2xxxEncoders have different return types, we return `sorted`
        outputs from the module, which is called directly. Additionally, we return the
        indices into the batch dimension required to restore the tensor to it's correct,
        unsorted order and the number of valid batch elements (i.e the number of elements
        in the batch which are not completely masked). This un-sorting and re-padding
        of the module outputs is left to the subclasses because their outputs have different
        types and handling them smoothly here is difficult.

        # Parameters

        module : `Callable[RnnInputs, RnnOutputs]`
            A function to run on the inputs, where
            `RnnInputs: [PackedSequence, Optional[RnnState]]` and
            `RnnOutputs: Tuple[Union[PackedSequence, torch.Tensor], RnnState]`.
            In most cases, this is a `torch.nn.Module`.
        inputs : `torch.Tensor`, required.
            A tensor of shape `(batch_size, sequence_length, embedding_size)` representing
            the inputs to the Encoder.
        mask : `torch.BoolTensor`, required.
            A tensor of shape `(batch_size, sequence_length)`, representing masked and
            non-masked elements of the sequence for each element in the batch.
        hidden_state : `Optional[RnnState]`, (default = `None`).
            A single tensor of shape (num_layers, batch_size, hidden_size) representing the
            state of an RNN with or a tuple of
            tensors of shapes (num_layers, batch_size, hidden_size) and
            (num_layers, batch_size, memory_size), representing the hidden state and memory
            state of an LSTM-like RNN.

        # Returns

        module_output : `Union[torch.Tensor, PackedSequence]`.
            A Tensor or PackedSequence representing the output of the Pytorch Module.
            The batch size dimension will be equal to `num_valid`, as sequences of zero
            length are clipped off before the module is called, as Pytorch cannot handle
            zero length sequences.
        final_states : `Optional[RnnState]`
            A Tensor representing the hidden state of the Pytorch Module. This can either
            be a single tensor of shape (num_layers, num_valid, hidden_size), for instance in
            the case of a GRU, or a tuple of tensors, such as those required for an LSTM.
        restoration_indices : `torch.LongTensor`
            A tensor of shape `(batch_size,)`, describing the re-indexing required to transform
            the outputs back to their original batch order.
        """
        # In some circumstances you may have sequences of zero length. `pack_padded_sequence`
        # requires all sequence lengths to be > 0, so remove sequences of zero length before
        # calling self._module, then fill with zeros.

        # First count how many sequences are empty.
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item()

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        (
            sorted_inputs,
            sorted_sequence_lengths,
            restoration_indices,
            sorting_indices,
        ) = sort_batch_by_length(inputs, sequence_lengths)

        # Now create a PackedSequence with only the non-empty, sorted sequences.
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs[:num_valid, :, :],
            sorted_sequence_lengths[:num_valid].data.tolist(),
            batch_first=True,
        )
        # Prepare the initial states.
        if not self.stateful:
            if hidden_state is None:
                initial_states: Any = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [
                    state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous()
                    for state in hidden_state
                ]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[
                    :, :num_valid, :
                ].contiguous()

        else:
            initial_states = self._get_initial_states(batch_size, num_valid, sorting_indices)

        # Actually call the module on the sorted PackedSequence.
        module_output, final_states = module(packed_sequence_input, initial_states)

        return module_output, final_states, restoration_indices

    def _get_initial_states(
        self, batch_size: int, num_valid: int, sorting_indices: torch.LongTensor
    ) -> Optional[RnnState]:
        """
        Returns an initial state for use in an RNN. Additionally, this method handles
        the batch size changing across calls by mutating the state to append initial states
        for new elements in the batch. Finally, it also handles sorting the states
        with respect to the sequence lengths of elements in the batch and removing rows
        which are completely padded. Importantly, this `mutates` the state if the
        current batch size is larger than when it was previously called.

        # Parameters

        batch_size : `int`, required.
            The batch size can change size across calls to stateful RNNs, so we need
            to know if we need to expand or shrink the states before returning them.
            Expanded states will be set to zero.
        num_valid : `int`, required.
            The batch may contain completely padded sequences which get removed before
            the sequence is passed through the encoder. We also need to clip these off
            of the state too.
        sorting_indices `torch.LongTensor`, required.
            Pytorch RNNs take sequences sorted by length. When we return the states to be
            used for a given call to `module.forward`, we need the states to match up to
            the sorted sequences, so before returning them, we sort the states using the
            same indices used to sort the sequences.

        # Returns

        This method has a complex return type because it has to deal with the first time it
        is called, when it has no state, and the fact that types of RNN have heterogeneous
        states.

        If it is the first time the module has been called, it returns `None`, regardless
        of the type of the `Module`.

        Otherwise, for LSTMs, it returns a tuple of `torch.Tensors` with shape
        `(num_layers, num_valid, state_size)` and `(num_layers, num_valid, memory_size)`
        respectively, or for GRUs, it returns a single `torch.Tensor` of shape
        `(num_layers, num_valid, state_size)`.
        """
        # We don't know the state sizes the first time calling forward,
        # so we let the module define what it's initial hidden state looks like.
        if self._states is None:
            return None

        # Otherwise, we have some previous states.
        if batch_size > self._states[0].size(1):
            # This batch is larger than the all previous states.
            # If so, resize the states.
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            # state has shape (num_layers, batch_size, hidden_size)
            for state in self._states:
                # This _must_ be inside the loop because some
                # RNNs have states with different last dimension sizes.
                zeros = state.new_zeros(state.size(0), num_states_to_concat, state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states

        elif batch_size < self._states[0].size(1):
            # This batch is smaller than the previous one.
            correctly_shaped_states = tuple(state[:, :batch_size, :] for state in self._states)
        else:
            correctly_shaped_states = self._states

        # At this point, our states are of shape (num_layers, batch_size, hidden_size).
        # However, the encoder uses sorted sequences and additionally removes elements
        # of the batch which are fully padded. We need the states to match up to these
        # sorted and filtered sequences, so we do that in the next two blocks before
        # returning the state/s.
        if len(self._states) == 1:
            # GRUs only have a single state. This `unpacks` it from the
            # tuple and returns the tensor directly.
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1, sorting_indices)
            return sorted_state[:, :num_valid, :].contiguous()
        else:
            # LSTMs have a state tuple of (state, memory).
            sorted_states = [
                state.index_select(1, sorting_indices) for state in correctly_shaped_states
            ]
            return tuple(state[:, :num_valid, :].contiguous() for state in sorted_states)

    def _update_states(
        self, final_states: RnnStateStorage, restoration_indices: torch.LongTensor
    ) -> None:
        """
        After the RNN has run forward, the states need to be updated.
        This method just sets the state to the updated new state, performing
        several pieces of book-keeping along the way - namely, unsorting the
        states and ensuring that the states of completely padded sequences are
        not updated. Finally, it also detaches the state variable from the
        computational graph, such that the graph can be garbage collected after
        each batch iteration.

        # Parameters

        final_states : `RnnStateStorage`, required.
            The hidden states returned as output from the RNN.
        restoration_indices : `torch.LongTensor`, required.
            The indices that invert the sorting used in `sort_and_run_forward`
            to order the states with respect to the lengths of the sequences in
            the batch.
        """
        # TODO(Mark): seems weird to sort here, but append zeros in the subclasses.
        # which way around is best?
        new_unsorted_states = [state.index_select(1, restoration_indices) for state in final_states]

        if self._states is None:
            # We don't already have states, so just set the
            # ones we receive to be the current state.
            self._states = tuple(state.data for state in new_unsorted_states)
        else:
            # Now we've sorted the states back so that they correspond to the original
            # indices, we need to figure out what states we need to update, because if we
            # didn't use a state for a particular row, we want to preserve its state.
            # Thankfully, the rows which are all zero in the state correspond exactly
            # to those which aren't used, so we create masks of shape (new_batch_size,),
            # denoting which states were used in the RNN computation.
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            # Masks for the unused states of shape (1, new_batch_size, 1)
            used_new_rows_mask = [
                (state[0, :, :].sum(-1) != 0.0).float().view(1, new_state_batch_size, 1)
                for state in new_unsorted_states
            ]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                # The new state is smaller than the old one,
                # so just update the indices which we used.
                for old_state, new_state, used_mask in zip(
                    self._states, new_unsorted_states, used_new_rows_mask
                ):
                    # zero out all rows in the previous state
                    # which _were_ used in the current state.
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    # The old state is larger, so update the relevant parts of it.
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                # The states are the same size, so we just have to
                # deal with the possibility that some rows weren't used.
                new_states = []
                for old_state, new_state, used_mask in zip(
                    self._states, new_unsorted_states, used_new_rows_mask
                ):
                    # zero out all rows which _were_ used in the current state.
                    masked_old_state = old_state * (1 - used_mask)
                    # The old state is larger, so update the relevant parts of it.
                    new_state += masked_old_state
                    new_states.append(new_state.detach())

            # It looks like there should be another case handled here - when
            # the current_state_batch_size < new_state_batch_size. However,
            # this never happens, because the states themeselves are mutated
            # by appending zeros when calling _get_inital_states, meaning that
            # the new states are either of equal size, or smaller, in the case
            # that there are some unused elements (zero-length) for the RNN computation.
            self._states = tuple(new_states)

    def reset_states(self, mask: torch.BoolTensor = None) -> None:
        """
        Resets the internal states of a stateful encoder.

        # Parameters

        mask : `torch.BoolTensor`, optional.
            A tensor of shape `(batch_size,)` indicating which states should
            be reset. If not provided, all states will be reset.
        """
        if mask is None:
            self._states = None
        else:
            # state has shape (num_layers, batch_size, hidden_size). We reshape
            # mask to have shape (1, batch_size, 1) so that operations
            # broadcast properly.
            mask_batch_size = mask.size(0)
            mask = mask.view(1, mask_batch_size, 1)
            new_states = []
            assert self._states is not None
            for old_state in self._states:
                old_state_batch_size = old_state.size(1)
                if old_state_batch_size != mask_batch_size:
                    raise ValueError(
                        f"Trying to reset states using mask with incorrect batch size. "
                        f"Expected batch size: {old_state_batch_size}. "
                        f"Provided batch size: {mask_batch_size}."
                    )
                new_state = ~mask * old_state
                new_states.append(new_state.detach())
            self._states = tuple(new_states)



class Seq2SeqEncoder(_EncoderBase):
    """
    A `Seq2SeqEncoder` is a `Module` that takes as input a sequence of vectors and returns a
    modified sequence of vectors.  Input shape : `(batch_size, sequence_length, input_dim)`; output
    shape : `(batch_size, sequence_length, output_dim)`.

    We add two methods to the basic `Module` API: `get_input_dim()` and `get_output_dim()`.
    You might need this if you want to construct a `Linear` layer using the output of this encoder,
    or to raise sensible errors for mis-matching input dimensions.
    """

    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2SeqEncoder`. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this `Seq2SeqEncoder`.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    def is_bidirectional(self) -> bool:
        """
        Returns `True` if this encoder is bidirectional.  If so, we assume the forward direction
        of the encoder is the first half of the final dimension, and the backward direction is the
        second half.
        """
        raise NotImplementedError

class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the first one as a single output.  This
    wrapper pulls out that output, and adds a `get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from `get_output_dim`.

    In order to be wrapped with this wrapper, a class must have the following members:

        - `self.input_size: int`
        - `self.hidden_size: int`
        - `def forward(inputs: PackedSequence, hidden_state: torch.Tensor) ->
          Tuple[PackedSequence, torch.Tensor]`.
        - `self.bidirectional: bool` (optional)

    This is what pytorch's RNN's look like - just make sure your class looks like those, and it
    should work.

    Note that we *require* you to pass a binary mask of shape (batch_size, sequence_length)
    when you call this module, to avoid subtle bugs around masking.  If you already have a
    `PackedSequence` you can pass `None` as the second parameter.

    We support stateful RNNs where the final state from each batch is used as the initial
    state for the subsequent batch by passing `stateful=True` to the constructor.
    """

    def __init__(self, module: torch.nn.Module, stateful: bool = False) -> None:
        super().__init__(stateful)
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

        try:
            self._is_bidirectional = self._module.bidirectional
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._num_directions

    def is_bidirectional(self) -> bool:
        return self._is_bidirectional

    def forward(
        self, inputs: torch.Tensor, mask: torch.BoolTensor, hidden_state: torch.Tensor = None
    ) -> torch.Tensor:

        if self.stateful and mask is None:
            raise ValueError("Always pass a mask with stateful RNNs.")
        if self.stateful and hidden_state is not None:
            raise ValueError("Stateful RNNs provide their own initial hidden_state.")

        if mask is None:
            return self._module(inputs, hidden_state)[0]

        batch_size, total_sequence_length = mask.size()

        packed_sequence_output, final_states, restoration_indices = self.sort_and_run_forward(
            self._module, inputs, mask, hidden_state
        )

        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

        num_valid = unpacked_sequence_tensor.size(0)
        # Some RNNs (GRUs) only return one state as a Tensor.  Others (LSTMs) return two.
        # If one state, use a single element list to handle in a consistent manner below.
        if not isinstance(final_states, (list, tuple)) and self.stateful:
            final_states = [final_states]

        # Add back invalid rows.
        if num_valid < batch_size:
            _, length, output_dim = unpacked_sequence_tensor.size()
            zeros = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, length, output_dim)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)

            # The states also need to have invalid rows added back.
            if self.stateful:
                new_states = []
                for state in final_states:
                    num_layers, _, state_dim = state.size()
                    zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                    new_states.append(torch.cat([state, zeros], 1))
                final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2SeqEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.new_zeros(
                batch_size, sequence_length_difference, unpacked_sequence_tensor.size(-1)
            )
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

        if self.stateful:
            self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

class LstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "lstm".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False,
    ):
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module, stateful=stateful)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
         pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
             output = pos_encoder(x)
        """

        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)



class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.05,vocab_size = None):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers,batch_first=True)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.linout = nn.Linear(ninp, vocab_size - 1)
        self.logit = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ninp, ninp * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ninp * 2, vocab_size - 1)
        )
        self.embedding_layer = nn.Embedding(vocab_size, ninp)
        #self.init_weights()


    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.bias)
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def make_trg_mask(self,trg):
        N, trg_len, emb_dim = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len)))

        return trg_mask.to(self.device)

    def create_pad_mask(self, idx_seq, pad_idx):
        # idx_seq shape: (seq len, batch size)
        mask = idx_seq == pad_idx
        # mask shape: (batch size, seq len) <- PyTorch transformer wants this shape for mask
        return mask

    def make_trg_mask_pad(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask
    def forward(self, src, trg, has_mask=False, max_len=100,start_token = None, end_token=None, is_inference = False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
                mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        batch_size = src.shape[0]

        # start of sequence
        starts = torch.full(
            size=(batch_size,), fill_value=start_token, device=src.device).long()
        ends = torch.full(
            size=(batch_size,), fill_value=end_token, device=src.device).long()
        # embed_start
        emb = self.embedding_layer(starts)
        emb_end = self.embedding_layer(ends)
        #src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        trg = torch.cat((torch.reshape(starts, (starts.shape[0], 1)), trg), dim=-1)
        trg_mask = self._generate_square_subsequent_mask(trg.shape[1]).to(src.device)
        trg_mask_paded = self.create_pad_mask(trg, 0).to(src.device)
        if not is_inference:
            trg = self.embedding_layer(trg)
        else:
            trg = self.embedding_layer(trg)


        #output = self.encoder(self.pos_encoder(trg),mask=trg_mask,src_key_padding_mask=trg_mask_paded)
        output = self.encoder(src, mask=self.src_mask)

        log_probabilities = []
        entropies = []



        trg = self.decoder(self.pos_encoder(trg),output,tgt_mask=trg_mask, tgt_key_padding_mask=trg_mask_paded)
        dist = Categorical(logits=self.logit(trg))
        sample = dist.sample()
        #x = torch.cat((x,sample),-1)
        # append log prob
        log_probabilities = dist.log_prob(sample)
        #
        ## append entropy
        entropies = dist.entropy()

        #output = self.decoder(output)
        return sample, log_probabilities, entropies, output


    def inference(self, src, has_mask=False, max_len=100,start_token = None, end_token=None, is_inference = False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
                mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        batch_size = src.shape[0]
        starts = torch.full(
            size=(batch_size,), fill_value=start_token, device=src.device).long()
        trg_mask_paded = self.create_pad_mask(starts, 0).to(src.device)
        seq = torch.reshape(self.embedding_layer(starts),(batch_size,1,-1))
        zero_token_emb = self.embedding_layer(torch.full(             size=(batch_size,), fill_value=0, device=src.device).long())[0]
        #src = self.input_emb(src) * math.sqrt(self.ninp)
        #src = self.pos_encoder(src)

        output = self.encoder(src, mask=self.src_mask)

        sample = torch.reshape(starts,(batch_size,1))
        for i in range(max_len):
            trg = self.decoder(self.pos_encoder(seq), output,
                               tgt_mask=self._generate_square_subsequent_mask(seq.shape[1]).to(src.device))
            #trg = self.decoder(self.pos_encoder(seq),src,tgt_mask=self._generate_square_subsequent_mask(seq.shape[1]).to(src.device))
            seq = torch.cat((seq,torch.reshape(trg[:,-1],(batch_size,1,-1))),dim=1)
        dist = Categorical(logits=self.logit(trg))
        x = dist.sample()
        end_pos = (x == end_token).float().argmax(dim=1).cpu()

        # sequence length is end token position + 1
        seq_lengths = end_pos - (x[:, 0] == end_token).int().cpu()

        # if end_pos = 0 => put seq_length = max_len
        seq_lengths.masked_fill_(seq_lengths == 0, max_len)
        seq_lengths.masked_fill_(seq_lengths == -1, 0).int()

        # select up to length
        _x = []

        for x_i, length in zip(x, seq_lengths):
            _x.append(x_i[:length+1])

        x = torch.nn.utils.rnn.pad_sequence(
            _x, batch_first=True, padding_value=-1)

        x = x + 1  # add padding token
        return x