import torch
import os
import pickle


class SaverLoader:
    def __init__(self, config, model, load_only=False):
        self.model = model
        self.config = config
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        if load_only:
            assert(self.config.loadname is not None)
        else:
            self.save_path = self.config.checkpoint_dir
            self.current_saved_model = self.init_list(path=self.save_path)

    def init_list(self, path):
        # making the initial weight list
        weight_list = []
        self.list_file = os.path.join(path, "weight.list")

        if os.path.isfile(self.list_file):
            with open(self.list_file, 'r') as f:
                weight_list = [int(x) for x in f]
        return weight_list

    def load_weights(self, optimizer=None):
        """Loads weights for the entire model or part of it,
           if optimizer is None, it implies I am loading for the part
        """
        if self.config.total_init:
            print("TOTAL INIT")
            print(self.config.total_init)
            start_iter = self.load(self.config.total_init, self.model, optimizer)
            if start_iter:
                print("loaded full model. resuming from iter %d" % start_iter)
            else:
                print("could not find a full model. starting from scratch")
        else:
            start_iter = 0
            inits = {"featnet": self.config.feat_init,
                     "viewnet": self.config.view_init,
                     "visnet": self.config.vis_init,
                     "flownet": self.config.flow_init,
                     "embnet2d": self.config.emb2D_init,
                     # "embnet3d": hyp.emb3D_init, # no params here really
                     "inpnet": self.config.inp_init,
                     "egonet": self.config.ego_init,
                     "occnet": self.config.occ_init,
                     "context_net": self.config.touch_forward_init,
                     "backbone_2D": self.config.touch_feat_init,
                     "key_touch_featnet": self.config.touch_feat_init,
                     "key_context_net": self.config.touch_forward_init
            }

            for part, init in list(inits.items()):
                if init:
                    if part == 'featnet':
                        model_part = self.model.featnet
                    elif part == 'viewnet':
                        model_part = self.model.viewnet
                    elif part == 'occnet':
                        model_part = self.model.occnet
                    elif part == 'embnet2d':
                        print('I should not come here in embnet2D')
                        from IPython import embed; embed()
                        model_part = self.model.embnet2D
                    elif part == 'context_net':
                        model_part = self.model.context_net
                    elif part == 'backbone_2D':
                        model_part = self.model.backbone_2D
                    elif part == 'key_touch_featnet':
                        model_part = self.model.key_touch_featnet
                    elif part == 'key_context_net':
                        model_part = self.model.key_context_net
                    else:
                        assert(False)
                    iter = load_part(model_part, part, init)
                    if iter:
                        print("loaded %s at iter %d" % (init, iter))
                    else:
                        print("could not find a checkpoint for %s" % init)
        if self.config.reset_iter:
            start_iter = 0
        return start_iter

    def save(self, step, optimizer):
        model_name = "model-%d.pth"%(step)
        path = os.path.join(self.save_path, model_name)

        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'local_variables': self.model.save_local_variables()
            }, path)
        print("Saved a checkpoint: %s"%(path))
        self.update_list(step)

    def remove_saved_model(self, step):
        model_name = "model-%d.pth"%(step)
        filename = os.path.join(self.save_path, model_name)
        if os.path.isfile(filename):
            os.remove(filename)
        #for layer in self.model.layers:
        #    filename = os.path.join(self.save_path, f"{step}-" + layer.name + '.h5')
        #    if os.path.isfile(filename):
        #        os.remove(filename)

    def save_config(self):
        config = self.config.__dict__
        for key in config:
            if type(config[key]) not in [str, list, dict, float, bool, int, type(None)] or '__' in key:
                config[key] = None

        filename = os.path.join(self.save_path, 'config.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(config, f)

    def update_list(self, step):
        """
        only keep the most recent 5 models
        """
        self.current_saved_model.append(step)
        if len(self.current_saved_model) > 5:
            oldest_step = self.current_saved_model.pop(0)
            self.remove_saved_model(oldest_step)
        with open(self.list_file, "w") as f:
            f.write("\n".join([str(x) for x in self.current_saved_model]))


    def load(self, model_name, model, optimizer):
        print("reading full checkpoint...")
        #checkpoint_dir = os.path.join("checkpoints/", model_name)
        step = 0

        if self.config.loadname == None: # didin't load from something
            checkpoint_dir =  self.save_path #os.path.join("checkpoints/", model_name)

            if len(self.current_saved_model) > 0:
                step = self.current_saved_model[-1]
                model_name = 'model-%d.pth'%(step)
                path = os.path.join(checkpoint_dir, model_name)
                print("...found checkpoint %s"%(path))
            else: # no saved weight, start from 0
                step = 0
                path = None
            #if len(ckpt_names) > 0:
        else:
            checkpoint_dir = self.config.loadname['model']
            if not os.path.exists(checkpoint_dir):
                print("...ain't no full checkpoint here!")
                assert(1==2)
            else:
                if "-" in checkpoint_dir.split("/")[-1]: # having specified the step
                    path = checkpoint_dir
                    step = int(checkpoint_dir.split("/")[-1].split("-")[1][:-4])
                else:
                    with open(os.path.join(checkpoint_dir, "weight.list"), 'r') as f:
                        weight_list = [int(x) for x in f]
                    step = weight_list[-1]
                
                    #ckpt_names = os.listdir(checkpoint_dir)
                    #steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
                    #if len(ckpt_names) > 0:
                    #step = max(steps)
                    model_name = 'model-%d.pth'%(step)
                    path = os.path.join(checkpoint_dir, model_name)
                    print("...found checkpoint %s"%(path))



        if path:
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("...no optimizer here!")
            checkpoint["local_variables"]
            model.__dict__.update(checkpoint["local_variables"])

        return step

    def load_part(self, model, part, init):
        """Note This only load the last step model, you have to do early stopping if you wish
           to not do it."""
        print("reading %s checkpoint..." % part)
        init_dir = os.path.join("checkpoints", init)
        print(init_dir)
        step = 0
        if not os.path.exists(init_dir):
            print("...ain't no %s checkpoint here!"%(part))
        else:
            ckpt_names = os.listdir(init_dir)
            steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
            if len(ckpt_names) > 0:
                step = max(steps)  # TODO: Ideally I should have a choice for which model I want to load
                model_name = 'model-%d.pth'%(step)
                path = os.path.join(init_dir, model_name)
                print("...found checkpoint %s"%(path))

                checkpoint = torch.load(path, map_location=self.device)
                model_state_dict = model.state_dict()
                # print(model_state_dict.keys())
                for load_para_name, para in checkpoint['model_state_dict'].items():
                    model_para_name = load_para_name[len(part)+1:]
                    # print(model_para_name, load_para_name)
                    if part+"."+model_para_name != load_para_name:
                        continue
                    else:
                        print(f'copying weights for {model_para_name}')
                        model_state_dict[model_para_name].copy_(para.data)
                    #model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                print("...ain't no %s checkpoint here!"%(part))
        return step