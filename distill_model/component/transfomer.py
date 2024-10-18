class Transformer(Cell):
    r"""
        Transformer module including encoder and decoder. The difference with the original implements is the module use
        the residual addition before the layer normalization. And the default hidden act is `gelu`.
        The details can be found in `Attention is all you need <https://arxiv.org/pdf/1706.03762v5.pdf>`_.

        Note:
            This is an experimental interface that is subject to change or deletion.

        Args:
            hidden_size(int): The hidden size of the input.
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
            src_seq_length(int): The seq_length of the encoder's input tensor.
            tgt_seq_length(int): The seq_length of the decoder's input tensor.
            encoder_layers(int): The layers of the `TransformerEncoderLayer`. Default 3.
            decoder_layers(int): The layers of the `TransformerDecoderLayer`. Default 3.
            num_heads(int): The number of the heads. Default: 2.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see the examples of the
                class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
            post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
            layernorm_compute_type(dtype.Number): The computation type of the layernorm.
                Should be dtype.float32 or dtype.float16. Default dtype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be dtype.float32 or dtype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be dtype.float32 or dtype.float16. Default dtype.float32.
            lambda_func: A function can determine the fusion index, pipeline stages and recompute attribute. If the user
                wants to determine the pipeline stage and gradient aggregation fusion, the user can pass a function
                that accepts `network`, `layer_id`, `offset`, `parallel_config`, `layers`. The `network(Cell)`
                represents the transformer block, `layer_id(int)` means the layer index for the current module, counts
                from zero, `offset(int)` means the layer_index needs an offset, if there are other modules in the net.
                The default setting for the pipeline is: `(layer_id + offset) // ((encoder_layers + decoder_layers)
                / pipeline_stage)`. Default None.
            use_past(bool): Use the past state to compute, used for incremental prediction. Default False.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
                with default values. Please see `MoEConfig`.
            parallel_config(TransformerOpParallelConfig): The parallel configure. Default `default_transformer_config`,
                an instance of `TransformerOpParallelConfig` with default args.

        Inputs:
            - **encoder_inputs** (Tensor) - The input tensor with shape [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size].
            - **encoder_masks** (Tensor) - The attention mask for decoder with shape
              [batch_size, seq_length, seq_length] or None. None means there will be no mask in softmax computation
              in self attention of the encoder module.
            - **decoder_inputs** (Tensor) - The output of the encoder with shape [batch_size, seq_length, hidden_size]
              or [batch_size * seq_length, hidden_size], this should be none if the decoder layer is 0.
            - **decoder_masks** (Tensor) - The attention mask for decoder with shape
              [batch_size, seq_length, seq_length] or None. None means there will be no mask in softmax computation
              in self attention of the decoder module.
            - **memory_mask** (Tensor) - The memory mask of the cross attention with shape [batch, tgt_seq_length,
              src_seq_length]
              where tgt_seq_length is the length of the decoder. The output of the encoder with shape [batch_size,
              seq_length, hidden_size], this should be none if the decoder layer is 0 or the user wants no mask.
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `encoder_layer_present`, `decoder_layer_present`, `accum_loss`)

            - **output** (Tensor) - If there is only encoder, the output logit of the encoder layer. The shape is
              [batch, src_seq_length, hidden_size] or [batch * src_seq_length, hidden_size], if there are encoder and
              decoders, the output is from the decoder layer. The shape is [batch, tgt_seq_length, hidden_size] or
              [batch * tgt_seq_length, hidden_size].
            - **encoder_layer_present** (Tuple) - A tuple with size of num_layers, where each tuple is the tensor the
              projected key and value vector in self attention with shape ((batch_size, num_heads, size_per_head,
              src_seq_length), (batch_size, num_heads, src_seq_length, size_per_head)).
            - **decoder_layer_present** (Tuple) - A tuple with size of num_layers, where each tuple is the tensor
              of the projected key and value vector in self attention with shape ((batch_size, num_heads, size_per_head,
              tgt_seq_length), (batch_size, num_heads, tgt_seq_length, size_per_head)), and the
              projected key and value vector in cross attention with shape
              ((batch_size, num_heads, size_per_head, src_seq_length),
              (batch_size, num_heads, src_seq_length, size_per_head)). If the decoder is not set, the
              returned value will be None.
            - **accum_loss** (Tensor) - A Tensor indicates an auxiliary loss to minimize the mean square of the data
              part routed to each expert, and only returned if the number of experts is greater than 1.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import dtype as mstype
            >>> from mindformers.modules.transformer import Transformer
            >>> from mindspore import Tensor
            >>> model = Transformer(batch_size=2, encoder_layers=1, decoder_layers=2, hidden_size=64,
            ...                     ffn_hidden_size=64, src_seq_length=20, tgt_seq_length=10)
            >>> encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
            >>> encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
            >>> decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
            >>> decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
            >>> memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
            >>> output, en_past, de_past = model(encoder_input_value, encoder_input_mask, decoder_input_value,
            ...                                  decoder_input_mask, memory_mask)
            >>> print(output.shape)
            (2, 10, 64)
            >>> print(len(en_past))
            1
            >>> print(len(de_past))
            2
            >>> print(en_past[0][0].shape)
            (2, 2, 32, 20)
            >>> print(en_past[0][1].shape)
            (2, 2, 20, 32)
            >>> print(de_past[0][0].shape)
            (2, 2, 32, 10)
            >>> print(de_past[0][1].shape)
            (2, 2, 10, 32)
            >>> print(de_past[0][2].shape)
            (2, 2, 32, 20)
            >>> print(de_past[0][3].shape)
            (2, 2, 20, 32)
    """

    @_LogActionOnce(m_logger=logger, key='Transformer',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                src_seq_length=Validator.check_positive_int,
                                encoder_layers=Validator.check_positive_int,
                                decoder_layers=Validator.check_non_negative_int,
                                tgt_seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                post_layernorm_residual=Validator.check_bool,
                                layernorm_compute_type=_valid_value_checks([mstype.float32,
                                                                            mstype.float16, mstype.bfloat16],
                                                                           "Transformer"),
                                softmax_compute_type=_valid_value_checks([mstype.float32,
                                                                          mstype.float16, mstype.bfloat16],
                                                                         "Transformer"),
                                param_init_type=_valid_value_checks([mstype.float32,
                                                                     mstype.float16, mstype.bfloat16], "Transformer"),
                                parallel_config=_valid_type_checks([TransformerOpParallelConfig], "Transformer"),
                                use_past=Validator.check_bool)
    def __init__(self,
                 hidden_size,
                 batch_size,
                 ffn_hidden_size,
                 src_seq_length,
                 tgt_seq_length,
                 encoder_layers=3,
                 decoder_layers=3,
                 num_heads=2,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act='gelu',
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config):
        super(Transformer, self).__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.use_past = use_past
            if encoder_layers <= 0 < decoder_layers:
                raise ValueError(f"Transformer doest support encoder layer {encoder_layers} and decoder"
                                 f"layer {decoder_layers}, please use TransformerDecoder")
            if encoder_layers > 0 and decoder_layers > 0 and use_past:
                raise ValueError(f"The {self.cls_name} with encoder and decoder does not support use_past=True.")
            # The shard setting of Transformer is set within the TransformerEncoderLayer
            if not lambda_func:
                lambda_func = _get_lambda_func(total_layer=encoder_layers + decoder_layers)
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            if encoder_layers > 0:
                self.encoder = TransformerEncoder(num_layers=encoder_layers,
                                                  batch_size=batch_size,
                                                  hidden_size=hidden_size,
                                                  ffn_hidden_size=ffn_hidden_size,
                                                  num_heads=num_heads,
                                                  seq_length=src_seq_length,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  hidden_act=hidden_act,
                                                  layernorm_compute_type=layernorm_compute_type,
                                                  softmax_compute_type=softmax_compute_type,
                                                  post_layernorm_residual=post_layernorm_residual,
                                                  param_init_type=param_init_type,
                                                  lambda_func=lambda_func,
                                                  use_past=use_past,
                                                  moe_config=moe_config,
                                                  parallel_config=parallel_config)
            else:
                self.encoder = None

            # Offset is needed as the encoder has consumed some flags.
            # so the decoder need to increase the flags based on the encoder layer
            self.decoder = None
            if decoder_layers > 0:
                self.decoder = TransformerDecoder(num_layers=decoder_layers,
                                                  batch_size=batch_size,
                                                  hidden_size=hidden_size,
                                                  ffn_hidden_size=ffn_hidden_size,
                                                  num_heads=num_heads,
                                                  src_seq_length=src_seq_length,
                                                  tgt_seq_length=tgt_seq_length,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  hidden_act=hidden_act,
                                                  post_layernorm_residual=post_layernorm_residual,
                                                  layernorm_compute_type=layernorm_compute_type,
                                                  softmax_compute_type=softmax_compute_type,
                                                  lambda_func=lambda_func,
                                                  use_past=use_past,
                                                  param_init_type=param_init_type,
                                                  offset=encoder_layers,
                                                  moe_config=moe_config,
                                                  parallel_config=parallel_config)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.use_past = use_past
            if encoder_layers <= 0 < decoder_layers:
                raise ValueError(f"Transformer doest support encoder layer {encoder_layers} and decoder"
                                 f"layer {decoder_layers}, please use TransformerDecoder")
            if encoder_layers > 0 and decoder_layers > 0 and use_past:
                raise ValueError(f"The {self.cls_name} with encoder and decoder does not support use_past=True.")
            logger.warning("For parallel mode, sharding propagation is recommended, you can use it by setting "
                           "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                           "search_mode=\"sharding_propagation\")' and "
                           "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'")
            # The shard setting of Transformer is set within the TransformerEncoderLayer
            if not lambda_func:
                lambda_func = _get_lambda_func(total_layer=encoder_layers + decoder_layers)
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            if encoder_layers > 0:
                self.encoder = TransformerEncoder(num_layers=encoder_layers,
                                                  batch_size=batch_size,
                                                  hidden_size=hidden_size,
                                                  ffn_hidden_size=ffn_hidden_size,
                                                  num_heads=num_heads,
                                                  seq_length=src_seq_length,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  hidden_act=hidden_act,
                                                  layernorm_compute_type=layernorm_compute_type,
                                                  softmax_compute_type=softmax_compute_type,
                                                  post_layernorm_residual=post_layernorm_residual,
                                                  param_init_type=param_init_type,
                                                  lambda_func=lambda_func,
                                                  use_past=use_past,
                                                  moe_config=moe_config,
                                                  parallel_config=parallel_config)
            else:
                self.encoder = None

            # Offset is needed as the encoder has consumed some flags.
            # so the decoder need to increase the flags based on the encoder layer
            self.decoder = None
            if decoder_layers > 0:
                self.decoder = TransformerDecoder(num_layers=decoder_layers,
                                                  batch_size=batch_size,
                                                  hidden_size=hidden_size,
                                                  ffn_hidden_size=ffn_hidden_size,
                                                  num_heads=num_heads,
                                                  src_seq_length=src_seq_length,
                                                  tgt_seq_length=tgt_seq_length,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  hidden_act=hidden_act,
                                                  post_layernorm_residual=post_layernorm_residual,
                                                  layernorm_compute_type=layernorm_compute_type,
                                                  softmax_compute_type=softmax_compute_type,
                                                  lambda_func=lambda_func,
                                                  use_past=use_past,
                                                  param_init_type=param_init_type,
                                                  offset=encoder_layers,
                                                  moe_config=moe_config,
                                                  parallel_config=parallel_config)
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, encoder_inputs,
                  encoder_masks,
                  decoder_inputs=None,
                  decoder_masks=None,
                  memory_mask=None,
                  init_reset=True,
                  batch_valid_length=None):
        """process process"""
        encoder_output = None
        output = None
        encoder_layer_present = None
        decoder_layer_present = None
        accum_loss = self.aux_loss
        if self.encoder is not None:
            if self.use_moe:
                encoder_output, encoder_layer_present, encoder_aux_loss = self.encoder(encoder_inputs, encoder_masks,
                                                                                       init_reset, batch_valid_length)
                accum_loss = self.add(accum_loss, encoder_aux_loss)
            else:
                encoder_output, encoder_layer_present = self.encoder(encoder_inputs, encoder_masks, init_reset,
                                                                     batch_valid_length)
            output = encoder_output

        if self.decoder is not None:
            # decoder mask should be created outside of the model
            if self.use_moe:
                decoder_output, decoder_layer_present, decoder_aux_loss = self.decoder(decoder_inputs, decoder_masks,
                                                                                       encoder_output, memory_mask,
                                                                                       init_reset, batch_valid_length)
                accum_loss = self.add(accum_loss, decoder_aux_loss)
            else:
                decoder_output, decoder_layer_present = self.decoder(decoder_inputs,
                                                                     decoder_masks,
                                                                     encoder_output,
                                                                     memory_mask, init_reset,
                                                                     batch_valid_length)
            output = decoder_output
        if self.use_moe:
            return output, encoder_layer_present, decoder_layer_present, accum_loss
        return output, encoder_layer_present, decoder_layer_present