--
-- Modified from https://github.com/karpathy/char-rnn/blob/master/util/CharSplitLMMinibatchLoader.lua
-- which is based on https://github.com/oxford-cs-ml-2015/practical6
--

local WordSplitLMMinibatchLoader = {}
WordSplitLMMinibatchLoader.__index = WordSplitLMMinibatchLoader

function WordSplitLMMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions, in_wordvecfile, in_wordveclen)
    WordSplitLMMinibatchLoader.wordvecfile = in_wordvecfile
    WordSplitLMMinibatchLoader.wordveclen = in_wordveclen
    -- split_fractions is e.g. {0.9, 0.05, 0.05}
    local self = {}

    setmetatable(self, WordSplitLMMinibatchLoader)

    local input_file_tr_Q = path.join(data_dir, 'train', 'questions.txt')
    local input_file_tr_A = path.join(data_dir, 'train', 'answers.txt')
    local input_file_tr_T = path.join(data_dir, 'train', 'types.txt')
    local input_file_tr_I = path.join(data_dir, 'train', 'img_ids.txt')
    local input_file_tt_Q = path.join(data_dir, 'test',  'questions.txt')
    local input_file_tt_A = path.join(data_dir, 'test',  'answers.txt')
    local input_file_tt_T = path.join(data_dir, 'test',  'types.txt')
    local input_file_tt_I = path.join(data_dir, 'test',  'img_ids.txt')
    local data_file = path.join(data_dir, 'data.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(data_file)) then
        -- prepro files do not exist, generate them
        print('data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local data_attr = lfs.attributes(data_file)
        if (lfs.attributes(input_file_tr_Q).modification > data_attr.modification 
            or lfs.attributes(input_file_tr_A).modification > data_attr.modification
            or lfs.attributes(input_file_tr_T).modification > data_attr.modification
            or lfs.attributes(input_file_tr_I).modification > data_attr.modification
            or lfs.attributes(input_file_tt_Q).modification > data_attr.modification
            or lfs.attributes(input_file_tt_A).modification > data_attr.modification
            or lfs.attributes(input_file_tt_T).modification > data_attr.modification
            or lfs.attributes(input_file_tt_I).modification > data_attr.modification) then
            print('data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file in ' .. data_dir .. '...')
        WordSplitLMMinibatchLoader.text_to_tensor(data_dir, data_file)
    end

    print('loading data files...')
    local data = torch.load(data_file)
    self.vocab_mapping_Q = data.vocab_mapping_Q
    self.vocab_mapping_A = data.vocab_mapping_A

    -- cut off the end so that it divides evenly
    local len
    len = #data.data_tr_Q
    len = len[1]
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data.data_tr_Q = data.data_tr_Q:sub(1, batch_size * seq_length 
            * math.floor(len / (batch_size * seq_length)))
        len_tr_A = #data.data_tr_Q
        len_tr_A = len_tr_A[1]/seq_length
        data.data_tr_A = data.data_tr_A:sub(1, len_tr_A)
        data.data_tr_T = data.data_tr_T:sub(1, len_tr_A)
        data.data_tr_I = data.data_tr_I:sub(1, len_tr_A)
    end
    len = #data.data_tt_Q
    len = len[1]
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data.data_tt_Q = data.data_tt_Q:sub(1, batch_size * seq_length 
            * math.floor(len / (batch_size * seq_length)))
        len_tt_A = #data.data_tt_Q
        len_tt_A = len_tt_A[1]/seq_length
        data.data_tt_A = data.data_tt_A:sub(1, len_tt_A)
        data.data_tt_T = data.data_tt_T:sub(1, len_tt_A)
        data.data_tt_I = data.data_tt_I:sub(1, len_tt_A)
    end

    -- count vocab
    self.vocab_size_Q = 0
    for _ in pairs(self.vocab_mapping_Q) do 
        self.vocab_size_Q = self.vocab_size_Q + 1 
    end
    self.vocab_size_A = 0
    for _ in pairs(self.vocab_mapping_A) do 
        self.vocab_size_A = self.vocab_size_A + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local xdata_tr = data.data_tr_Q
    local ydata_tr = data.data_tr_A:view(-1,1)
    ydata_tr = torch.repeatTensor(ydata_tr,1,seq_length)
    ydata_tr = ydata_tr:view(-1)
    local x_batches_tr = xdata_tr:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches (only train)
    local y_batches_tr = ydata_tr:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches (only train)
    assert(#x_batches_tr == #y_batches_tr)

    local xdata_tt = data.data_tt_Q
    local ydata_tt = data.data_tt_A:view(-1,1)
    ydata_tt = torch.repeatTensor(ydata_tt,1,seq_length)
    ydata_tt = ydata_tt:view(-1)
    local x_batches_tt = xdata_tt:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches (only test)
    local y_batches_tt = ydata_tt:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches (only test)
    assert(#x_batches_tt == #y_batches_tt)

    self.x_batches = torch.cat(xdata_tr,xdata_tt):view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches (tr & tt)
    self.y_batches = torch.cat(ydata_tr,ydata_tt):view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches (tr & tt)

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    
    self.ntrain = math.floor(#x_batches_tr * split_fractions[1])
    self.nval = #x_batches_tr - self.ntrain
    self.ntest = #x_batches_tt
    self.nbatches = #self.x_batches
    assert(self.nbatches == self.ntrain+self.nval+self.ntest)

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function WordSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function WordSplitLMMinibatchLoader:next_batch(split_index)
    if WordSplitLMMinibatchLoader.wordvec == nil then
        WordSplitLMMinibatchLoader.wordvec = torch.load(WordSplitLMMinibatchLoader.wordvecfile)
    end
    assert(not (WordSplitLMMinibatchLoader.wordvec == nil), 'WordSplitLMMinibatchLoader.wordvec should not be nil')

    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val

    -- return wordvec's
    local tmp_x_batches = self.x_batches[ix]:float()
    local tmp_wordvec_batches = torch.Tensor(self.batch_size, self.seq_length, self.wordveclen)
    for i=1,self.batch_size do
        for j=1,self.seq_length do
            tmp_wordvec_batches[{i,j,{}}] = self.wordvec.index_to_emb[tmp_x_batches[i][j]]
        end
    end
    return tmp_wordvec_batches, self.y_batches[ix]
end

-- *** STATIC method ***
function WordSplitLMMinibatchLoader.text_to_tensor(in_textfolder, out_datafile)
    if WordSplitLMMinibatchLoader.wordvec == nil then
        WordSplitLMMinibatchLoader.wordvec = torch.load(WordSplitLMMinibatchLoader.wordvecfile)
    end
    assert(not (WordSplitLMMinibatchLoader.wordvec == nil), 'WordSplitLMMinibatchLoader.wordvec should not be nil')

    local timer = torch.Timer()
    local in_file_tr_Q = path.join(in_textfolder, 'train', 'questions.txt')
    local in_file_tr_A = path.join(in_textfolder, 'train', 'answers.txt')
    local in_file_tr_T = path.join(in_textfolder, 'train', 'types.txt')
    local in_file_tr_I = path.join(in_textfolder, 'train', 'img_ids.txt')
    local in_file_tt_Q = path.join(in_textfolder, 'test',  'questions.txt')
    local in_file_tt_A = path.join(in_textfolder, 'test',  'answers.txt')
    local in_file_tt_T = path.join(in_textfolder, 'test',  'types.txt')
    local in_file_tt_I = path.join(in_textfolder, 'test',  'img_ids.txt')

    print('loading text file...')
    local f
    local sentence
    local tot_len_tr_Q = 0
    local tot_len_tr_A = 0
    local tot_len_tt_Q = 0
    local tot_len_tt_A = 0

    ----------------------------------------------
    -- Create Vocabulary of Questions (tr & tt) --
    ----------------------------------------------
    local unordered_Q = {}
    -- Train Question
    f = io.open(in_file_tr_Q, "r")
    sentence = f:read()
    while sentence ~= nil do
        local sentence_len = 0
        for word in sentence:gmatch('%w+') do
            if not unordered_Q[word] then unordered_Q[word] = true end
            sentence_len = sentence_len + 1
        end
        tot_len_tr_Q = tot_len_tr_Q + sentence_len
        sentence = f:read()
    end
    f:close()
    -- Test Question
    f = io.open(in_file_tt_Q, "r")
    sentence = f:read()
    while sentence ~= nil do
        local sentence_len = 0
        for word in sentence:gmatch('%w+') do
            if not unordered_Q[word] then unordered_Q[word] = true end
            sentence_len = sentence_len + 1
        end
        tot_len_tt_Q = tot_len_tt_Q + sentence_len
        sentence = f:read()
    end
    f:close()

    --------------------------------------------
    -- Create Vocabulary of Answers (tr & tt) --
    --------------------------------------------
    local unordered_A = {}
    -- Train Answer
    f = io.open(in_file_tr_A, "r")
    sentence = f:read()
    while sentence ~= nil do
        if not unordered_A[sentence] then unordered_A[sentence] = true end
        tot_len_tr_A = tot_len_tr_A + 1
        sentence = f:read()
    end
    f:close()
    -- Test Answer
    f = io.open(in_file_tt_A, "r")
    sentence = f:read()
    while sentence ~= nil do
        if not unordered_A[sentence] then unordered_A[sentence] = true end
        tot_len_tt_A = tot_len_tt_A + 1
        sentence = f:read()
    end
    f:close()

    print('creating vocabulary mapping...')
    local ordered

    -----------------------------------------------
    -- Sort Into a Table (i.e. keys become 1..N) --
    -----------------------------------------------
    ordered = {}
    for word in pairs(unordered_Q) do ordered[#ordered + 1] = word end
    table.sort(ordered)
    -- invert `ordered` to create the word->int mapping
    local vocab_mapping_Q = {}
    for i, word in ipairs(ordered) do
        --vocab_mapping_Q[word] = i -- not use the sorted index
        local tmpIdx = WordSplitLMMinibatchLoader.wordvec.word_to_index[word]
        if tmpIdx == nil then
            vocab_mapping_Q[word] = WordSplitLMMinibatchLoader.wordvec.word_to_index['*unk*']
        else
            vocab_mapping_Q[word] = tmpIdx
        end
    end
    ordered = {}
    for word in pairs(unordered_A) do ordered[#ordered + 1] = word end
    table.sort(ordered)
    -- invert `ordered` to create the word->int mapping
    local vocab_mapping_A = {}
    for i, word in ipairs(ordered) do
        vocab_mapping_A[word] = i
    end

    ------------------------------------------
    -- Construct a Tensor with All The Data --
    ------------------------------------------
    print('putting data into tensor...')
    local currlen
    -- Train Question
    local data_tr_Q = torch.IntTensor(tot_len_tr_Q) -- store it into 1D first, then rearrange (ByteTensor<256, ShortTensor<32768, IntTensor)
    f = io.open(in_file_tr_Q, "r")
    currlen = 0
    sentence = f:read()
    while sentence ~= nil do
        for word in sentence:gmatch('%w+') do
            currlen = currlen + 1
            data_tr_Q[currlen] = vocab_mapping_Q[word]
        end
        sentence = f:read()
    end
    f:close()
    -- Test Question
    local data_tt_Q = torch.IntTensor(tot_len_tt_Q) -- store it into 1D first, then rearrange (ByteTensor<256, ShortTensor<32768, IntTensor)
    f = io.open(in_file_tt_Q, "r")
    currlen = 0
    sentence = f:read()
    while sentence ~= nil do
        for word in sentence:gmatch('%w+') do
            currlen = currlen + 1
            data_tt_Q[currlen] = vocab_mapping_Q[word]
        end
        sentence = f:read()
    end
    f:close()

    -- Train Answer
    local data_tr_A = torch.IntTensor(tot_len_tr_A) -- store it into 1D first, then rearrange (ByteTensor<256, ShortTensor<32768, IntTensor)
    f = io.open(in_file_tr_A, "r")
    currlen = 0
    sentence = f:read()
    while sentence ~= nil do
        currlen = currlen + 1
        data_tr_A[currlen] = vocab_mapping_A[sentence]
        sentence = f:read()
    end
    f:close()
    -- Test Answer
    local data_tt_A = torch.IntTensor(tot_len_tt_A) -- store it into 1D first, then rearrange (ByteTensor<256, ShortTensor<32768, IntTensor)
    f = io.open(in_file_tt_A, "r")
    currlen = 0
    sentence = f:read()
    while sentence ~= nil do
        currlen = currlen + 1
        data_tt_A[currlen] = vocab_mapping_A[sentence]
        sentence = f:read()
    end
    f:close()

    -- Train Type
    local data_tr_T = torch.IntTensor(tot_len_tr_A) -- store it into 1D first, then rearrange (ByteTensor<256, ShortTensor<32768, IntTensor)
    f = io.open(in_file_tr_T, "r")
    currlen = 0
    sentence = f:read()
    while sentence ~= nil do
        currlen = currlen + 1
        data_tr_T[currlen] = tonumber(sentence)
        sentence = f:read()
    end
    f:close()
    -- Test Type
    local data_tt_T = torch.IntTensor(tot_len_tt_A) -- store it into 1D first, then rearrange (ByteTensor<256, ShortTensor<32768, IntTensor)
    f = io.open(in_file_tt_T, "r")
    currlen = 0
    sentence = f:read()
    while sentence ~= nil do
        currlen = currlen + 1
        data_tt_T[currlen] = tonumber(sentence)
        sentence = f:read()
    end
    f:close()

    -- Train Image_ID
    local data_tr_I = torch.IntTensor(tot_len_tr_A) -- store it into 1D first, then rearrange (ByteTensor<256, ShortTensor<32768, IntTensor)
    f = io.open(in_file_tr_I, "r")
    currlen = 0
    sentence = f:read()
    while sentence ~= nil do
        currlen = currlen + 1
        data_tr_I[currlen] = tonumber(sentence)
        sentence = f:read()
    end
    f:close()
    -- Test Image_ID
    local data_tt_I = torch.IntTensor(tot_len_tt_A) -- store it into 1D first, then rearrange (ByteTensor<256, ShortTensor<32768, IntTensor)
    f = io.open(in_file_tt_I, "r")
    currlen = 0
    sentence = f:read()
    while sentence ~= nil do
        currlen = currlen + 1
        data_tt_I[currlen] = tonumber(sentence)
        sentence = f:read()
    end
    f:close()

    -- save output preprocessed files
    local data = {}
    data.vocab_mapping_Q = vocab_mapping_Q
    data.vocab_mapping_A = vocab_mapping_A
    data.data_tr_Q = data_tr_Q
    data.data_tr_A = data_tr_A
    data.data_tr_T = data_tr_T
    data.data_tr_I = data_tr_I
    data.data_tt_Q = data_tt_Q
    data.data_tt_A = data_tt_A
    data.data_tt_T = data_tt_T
    data.data_tt_I = data_tt_I
    print('saving ' .. out_datafile)
    torch.save(out_datafile, data)
end

return WordSplitLMMinibatchLoader

