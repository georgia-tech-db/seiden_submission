import torch
import torchvision
import tasti
import numpy as np
import os
from tqdm.autonotebook import tqdm

class Index:
    def __init__(self, config):
        self.config = config
        self.target_dnn_cache = tasti.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(train_or_test='train'),
            self.target_dnn_callback
        )
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='train')
        self.seed = self.config.seed
        self.rand = np.random.RandomState(seed=self.seed)
        self.cache_dir = self.get_cache_dir()

    def update_category(self, category):
        self.config.category = category
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='train')

    def get_cache_dir(self):
        raise NotImplementedError

    def get_num_workers(self):
        raise NotImplementedError

    def override_target_dnn_cache(self, target_dnn_cache, train_or_test='train'):
        '''
        This allows you to override tasti.utils.DNNOutputCache if you already have the target dnn
        outputs available somewhere. Returning a list or another 1-D indexable element will work.
        '''
        return target_dnn_cache
        
    def is_close(self, a, b):
        '''
        Define your notion of "closeness" as described in the paper between records "a" and "b".
        Return a Boolean.
        '''
        raise NotImplementedError
        
    def get_target_dnn_dataset(self, train_or_test='train'):
        '''
        Define your target_dnn_dataset under the condition of "train_or_test".
        Return a torch.utils.data.Dataset object.
        '''
        raise NotImplementedError
    
    def get_embedding_dnn_dataset(self, train_or_test='train'):
        '''
        Define your embedding_dnn_dataset under the condition of "train_or_test".
        Return a torch.utils.data.Dataset object.
        '''
        raise NotImplementedError
        
    def get_target_dnn(self):
        '''
        Define your Target DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError
        
    def get_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        raise NotImplementedError
        
    def get_pretrained_embedding_dnn(self):
        '''
        Define your Embeding DNN.
        Return a torch.nn.Module.
        '''
        return self.get_pretrained_embedding_dnn()
        
    def target_dnn_callback(self, target_dnn_output):
        '''
        Often times, you want to process the output of your target dnn into something nicer.
        This function is called everytime a target dnn output is computed and allows you to process it.
        If it is not defined, it will simply return the input.
        '''
        return target_dnn_output

    def do_mining(self):
        '''
        The mining step of constructing a TASTI. We will use an embedding dnn to compute embeddings
        of the entire dataset. Then, we will use FPFRandomBucketter to choose "distinct" datapoints
        that can be useful for triplet training.
        '''
        if self.config.do_mining:
            model = self.get_pretrained_embedding_dnn()
            try:
                model.cuda()
                model.eval()
            except:
                pass
            
            dataset = self.get_embedding_dnn_dataset(train_or_test='train')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.get_num_workers(),
                pin_memory=True
            )
            
            embeddings = []
            with torch.no_grad():

                for batch in tqdm(dataloader, desc='Embedding DNN'):
                    batch = batch.cuda()
                    output = model(batch).cpu()
                    embeddings.append(output)
                embeddings = torch.cat(embeddings, dim=0)
                embeddings = embeddings.numpy()
            
            bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_train, self.seed)
            reps, _, _ = bucketter.bucket(embeddings, self.config.max_k)
            self.training_idxs = reps
        else:
            print('number of training instances: ', self.config.nb_train)
            self.training_idxs = self.rand.choice(
                    len(self.get_embedding_dnn_dataset(train_or_test='train')),
                    size=self.config.nb_train,
                    replace=False
            )
            
    def do_training(self):
        '''
        Fine-tuning the embedding dnn via triplet loss. 
        '''
        if self.config.do_training:
            model = self.get_target_dnn()
            model.eval()
            model.cuda()
            
            for idx in tqdm(self.training_idxs, desc='Target DNN'):
                self.target_dnn_cache[idx]
            
            dataset = self.get_embedding_dnn_dataset(train_or_test='train')
            triplet_dataset = tasti.data.TripletDataset(
                dataset=dataset,
                target_dnn_cache=self.target_dnn_cache,
                list_of_idxs=self.training_idxs,
                is_close_fn=self.is_close,
                length=self.config.nb_training_its
            )
            dataloader = torch.utils.data.DataLoader(
                triplet_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.get_num_workers(),
                pin_memory=True
            )
            
            model = self.get_embedding_dnn()
            model.train()
            model.cuda()
            loss_fn = tasti.TripletLoss(self.config.train_margin)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train_lr)
            
            for anchor, positive, negative in tqdm(dataloader, desc='Training Step'):
                anchor = anchor.cuda(non_blocking=True)
                positive = positive.cuda(non_blocking=True)
                negative = negative.cuda(non_blocking=True)
                
                e_a = model(anchor)
                e_p = model(positive)
                e_n = model(negative)
                
                optimizer.zero_grad()
                loss = loss_fn(e_a, e_p, e_n)
                loss.backward()
                optimizer.step()

            save_directory = os.path.join(self.cache_dir, 'model.pt')
            torch.save(model.state_dict(), save_directory)
            self.embedding_dnn = model
        else:
            self.embedding_dnn = self.get_pretrained_embedding_dnn()
            
        del self.target_dnn_cache
        self.target_dnn_cache = tasti.DNNOutputCache(
            self.get_target_dnn(),
            self.get_target_dnn_dataset(train_or_test='test'),
            self.target_dnn_callback
        )
        self.target_dnn_cache = self.override_target_dnn_cache(self.target_dnn_cache, train_or_test='test')
        
            
    def do_infer(self):
        '''
        With our fine-tuned embedding dnn, we now compute embeddings for the entire dataset.
        '''
        save_directory = os.path.join(self.cache_dir, f'embeddings_{self.config.video_name}.npy')
        if self.config.do_infer:
            model = self.embedding_dnn
            model.eval()
            model.cuda()
            dataset = self.get_embedding_dnn_dataset(train_or_test='test')
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.get_num_workers(),
                pin_memory=True
            )

            embeddings = []
            for batch in tqdm(dataloader, desc='Inference'):
                try:
                    batch = batch.cuda()
                except:
                    pass
                with torch.no_grad():
                    output = model(batch).cpu()
                embeddings.append(output)  
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.numpy()


            np.save(save_directory, embeddings)
            self.embeddings = embeddings
        else:
            try:
                self.embeddings = np.load(save_directory)
            except:
                self.embeddings = None

    def do_bucketting(self, percent_fpf = 0.75):
        '''
        Given our embeddings, cluster them and store the reps, topk_reps, and topk_dists to finalize our TASTI.
        '''
        if self.embeddings is None:
            raise ValueError()

        if self.config.do_bucketting:
            bucketter = tasti.bucketters.FPFRandomBucketter(self.config.nb_buckets, self.seed)
            self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k, percent_fpf=percent_fpf)

            np.save( os.path.join(self.cache_dir, 'reps.npy'), self.reps)
            np.save( os.path.join(self.cache_dir, 'topk_reps.npy'), self.topk_reps)
            np.save( os.path.join(self.cache_dir, 'topk_dists.npy'), self.topk_dists)
        else:
            self.reps = np.load( os.path.join(self.cache_dir, '/reps.npy'))
            self.topk_reps = np.load( os.path.join(self.cache_dir, '/topk_reps.npy') )
            self.topk_dists = np.load( os.path.join(self.cache_dir, '/topk_dists.npy') )
            
    def crack(self):
        cache = self.target_dnn_cache.cache
        cached_idxs = []
        for idx in range(len(cache)):
            if cache[idx] != None:
                cached_idxs.append(idx)        
        cached_idxs = np.array(cached_idxs)
        bucketter = tasti.bucketters.CrackingBucketter(self.config.nb_buckets)
        self.reps, self.topk_reps, self.topk_dists = bucketter.bucket(self.embeddings, self.config.max_k, cached_idxs)

        np.save( os.path.join(self.cache_dir, '/reps.npy'), self.reps)
        np.save( os.path.join(self.cache_dir, '/topk_reps.npy'), self.topk_reps)
        np.save( os.path.join(self.cache_dir, '/topk_dists.npy'), self.topk_dists)

        
    def init(self, percent_fpf = 0.75):
        print('index initializing....')
        self.do_mining()
        print('mining complete!')
        self.do_training()
        print('training complete!')
        self.do_infer()
        print('inferring complete!')
        self.do_bucketting(percent_fpf=percent_fpf)
        print('bucketing complete!')
        
        for rep in tqdm(self.reps, desc='Target DNN Invocations'):
            self.target_dnn_cache[rep]
