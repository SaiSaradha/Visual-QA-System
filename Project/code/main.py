from Config_Manager import *

#import required libraries
import numpy as np
import json
import time
import spacy
import scipy
import scipy.io
from spacy.lang.en import English
import tensorflow as tf
from sklearn import preprocessing
from sklearn.externals import joblib
import operator
from itertools import islice
from keras.utils import np_utils
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp


class VQA_Class:

    def __init__(self):
        #set all hyperparameters of the network models here (common to all the models)
        # first get the unzipped required files
        self.train_ques_f = "../text_data/v2_OpenEnded_mscoco_train2014_questions.json"
        self.train_annot_f = "../text_data/v2_mscoco_train2014_annotations.json"
        self.val_ques_f = "../text_data/v2_OpenEnded_mscoco_val2014_questions.json"
        self.val_annot_f = "../text_data/v2_mscoco_val2014_annotations.json"
        self.test_ques_f = "../text_data/v2_OpenEnded_mscoco_test2015_questions.json"
        self.GloVe_f = "../pretrained/glove.840B.300d.txt"
        # Glove embedding - word vectors initialize:
        self.word_vec = {}
        self.token_getter = English()

        self.train_que_fn = "../text_data/train_que_mod.txt"
        self.train_queid_fn = "../text_data/train_queid_mod.txt"
        self.train_ans_fn = "../text_data/train_ans_mod.txt"
        self.train_quelen_fn = "../text_data/train_quelen_mod.txt"
        self.train_imgid_fn = "../text_data/train_imgid_mod.txt"

        self.val_que_fn = "../text_data/val_que_mod.txt"
        self.val_ans_fn = "../text_data/val_ans_mod.txt"
        self.val_imgid_fn = "../text_data/val_imgid_mod.txt"
            

        self.train_que_final = []
        self.train_ans_final = []
        self.train_img_final = []
        self.top_n = 1000

        self.vgg_model = "../image_data/coco/vgg_feats.mat"
        self.vgg_coco = "../model/coco_vgg_IDMap.txt"
        self.vgg_coco_map = {}
        self.num_cat = 0
        self.numeric_ans = preprocessing.LabelEncoder()
        self.img_features = []

        self.log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), '/tensorflow/final_project/logs/mlp_with_summaries')

        self.pred_fn = "../model/final_pred_results.txt"
        self.pred_flag_fn = "../model/final_pred_flag.txt"
		
		# image directory:
        img_dir = "F:/imgs/val2014/COCO_val2014_"
        

    def take(self, n, iterable):
        "Return first n items of the iterable as a list"
        return list(islice(iterable, n))
                
    def load_GloVe_embed(self):
        print("Loading Glove Model")
        count_diff = 0
        f = open(self.GloVe_f, 'r', encoding='utf-8')
        count_max = 100
        count = 0
        for line in f:
            #if count >= count_max:
                #break
            try:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                self.word_vec[word] = embedding
            except Exception:
                count_diff += 1
                print(len(splitLine))
                vec = splitLine[-300:]
                embedding = np.array([float(val) for val in vec])
                word_len = len(splitLine)-300
                word = splitLine[:word_len]
                word = ''.join(word)
                self.word_vec[word]=embedding
            if len(self.word_vec)%100000 == 0:
                print(len(self.word_vec))
            #count += 1
        #print(self.take(100, self.word_vec.items()))
        print ("Done. ",len(self.word_vec)," words loaded!")
        print("Total " + str(count_diff) + " number of words different structure")
        return

    def get_word_vec(self, word):
        try:
            if isinstance(word, str):
                word = word.lower()
            return self.word_vec[str(word)]
        except Exception as e:
            return None            

    def get_token(self, sentence):
        return self.token_getter(sentence)

    def get_vgg_coco_map(self):
        map_file = open(self.vgg_coco, 'r')
        for line in map_file:
            split_line = line.split()
            self.vgg_coco_map[split_line[0]] = int(split_line[1])
        return

    def write_data_file(self, choice):
        # now get the filenames where we will be writing the modified data
        if choice=='train':
            json_que = json.load(open(self.train_ques_f, 'r'))
            json_annot = json.load(open(self.train_annot_f, 'r'))
            que_fn = open(self.train_que_fn,'w')
            queid_fn = open(self.train_queid_fn,'w')
            ans_fn = open(self.train_ans_fn,'w')
            quelen_fn = open(self.train_quelen_fn,'w')
            imgid_fn = open(self.train_imgid_fn,'w')
            annot_loaded = json_annot['annotations']
        elif choice=='val':
            json_que = json.load(open(self.val_ques_f, 'r'))
            json_annot = json.load(open(self.val_annot_f, 'r'))
            que_fn = open(self.val_que_fn,'w')
            queid_fn = open("../text_data/val_queid_mod.txt",'w')
            ans_fn = open(self.val_ans_fn,'w')
            quelen_fn = open("../text_data/val_quelen_mod.txt",'w')
            imgid_fn = open(self.val_imgid_fn,'w')
            annot_loaded = json_annot['annotations']
        elif choice=='test':
            json_que = json.load(open(self.test_ques_f, 'r'))
            json_annot = None
            que_fn = open("../text_data/test_que_mod.txt",'w')
            ans_fn = None
            queid_fn = open("../text_data/test_queid_mod.txt",'w')
            quelen_fn = open("../text_data/test_quelen_mod.txt",'w')
            imgid_fn = open("../text_data/test_imgid_mod.txt",'w')
            annot_loaded = None
        else:
            raise RuntimeError('Incorrect choice for writing modified data')
        questions_loaded = json_que['questions']
        print("Writing " + str(choice))
        # we will write the foll- questions, question id, answers, image id for coco dataset, length of questions
        print(len(questions_loaded))
        for que_len, quest in zip(range(len(questions_loaded)),questions_loaded):
            que_fn.write(quest['question']+'\n')
            queid_fn.write(str(quest['question_id'])+'\n')
            if annot_loaded is not None:
                ans_temp = self.get_ans_per_choice(choice, annot_loaded[que_len]['answers'])
                ans_fn.write(ans_temp)
                ans_fn.write("\n")
            quelen_fn.write(str(len(self.get_token(quest['question'])))+'\n')
            imgid_fn.write(str(quest['image_id'])+'\n')
            if ((que_len/len(questions_loaded))*100)%100 == 0:
                print(str(que_len/len(questions_loaded)) + " % done")
        print("100% done")
        que_fn.close()
        queid_fn.close()
        quelen_fn.close()
        imgid_fn.close()
        if ans_fn is not None:
            ans_fn.close()
        return

    def get_max_ans(self, ans_temp):
        count_freq_ans = {}
        num_ans = 10
        for i in range(num_ans):
            count_freq_ans[ans_temp[i]['answer']] = 1
        for j in range(num_ans):
            count_freq_ans[ans_temp[j]['answer']] += 1
        return max(count_freq_ans.items(), key=operator.itemgetter(1))[0]
    
    def get_ans(self, ans_temp):
        num_ans = 10
        ans_for_que = []
        # each question has 10 answers (we can see in file that there are 10 answer ids)
        for i in range(num_ans):
            ans_for_que.append(ans_temp[i]['answer'])
        return ';'.join(ans_for_que)

    def get_ans_per_choice(self, choice, annot):
        if choice == 'train':
            return self.get_max_ans(annot)
        return self.get_ans(annot)

    def get_most_frequent_set(self):
        count_freq_ans = {}
        ans_fn = open(self.train_ans_fn,'r').read().splitlines()
        que_fn = open(self.train_que_fn,'r').read().splitlines()
        img_fn = open(self.train_imgid_fn,'r').read().splitlines()
        
        for ans in ans_fn:
            if ans in count_freq_ans:
                count_freq_ans[ans] += 1
            else:
                count_freq_ans[ans] = 1
        count_freq_sorted = sorted(count_freq_ans.items(), key=operator.itemgetter(1), reverse=True)
        count_freq_sorted = count_freq_sorted[0:self.top_n]
        final_ans, ans_count_freq = zip(*count_freq_sorted)
        for q, a, i in zip(que_fn, ans_fn, img_fn):
            if a in final_ans:
                self.train_que_final.append(q)
                self.train_ans_final.append(a)
                self.train_img_final.append(i)
        print(len(self.train_que_final))
        print(len(self.train_ans_final))
        print(len(self.train_img_final))
        return
    
    def get_data_in_format(self):
        # the data to be prepared are the foll - questions, question_ids, annotations, answers
        # and we need to do this for train, val and test
        self.write_data_file('train')
        self.write_data_file('val')
        self.write_data_file('test')
        # we select only the top 80% of answers and the questions that have the corresponding answers (rest are not considered)
        self.get_most_frequent_set()
        return

    def get_que_vec_lstm(self, q, num_cells):
        x_batch = np.zeros((len(q), num_cells, txtdim_mlp))
        for i in range(len(q)):
            tokens = self.token_getter(q[i])
            for j in range(len(tokens)):
                this_vec = self.get_word_vec(tokens[j])
                if this_vec is not None and j<num_cells:
                    x_batch[i,j,:] = this_vec
        return x_batch

    def get_que_vec(self, q):
        x_batch = np.zeros((len(q), txtdim_mlp))
        for i in range(len(q)):
            tokens = self.token_getter(q[i])
            for j in range(len(tokens)):
                this_vec = self.get_word_vec(tokens[j])
                if this_vec is not None:
                    x_batch[i,:] += this_vec
        return x_batch

    def get_ans_vec(self, a):
        ans_batch = self.numeric_ans.transform(a)
        num_class = self.numeric_ans.classes_.shape[0]
        ans_batch = np_utils.to_categorical(ans_batch, num_class)
        return ans_batch

    def get_img_vec(self, i):
        vgg_feat_dim = self.img_features.shape[0]
        img_batch = np.zeros((len(i), vgg_feat_dim))
        for j in range(len(i)):
            img_batch[j,:] = self.img_features[:,self.vgg_coco_map[i[j]]]
        return img_batch

    def get_input_for_nn_model(self, q, a, i):
        return self.get_que_vec(q), self.get_ans_vec(a), self.get_img_vec(i)

    def visual_results(self):
        val_pred = open(self.pred_fn, 'r').read().splitlines()
        flag_fn = open(self.pred_flag_fn, 'r').read().splitlines()
        ans_fn = open(self.val_ans_fn,'r').read().splitlines()
        que_fn = open(self.val_que_fn,'r').read().splitlines()
        img_fn = open(self.val_imgid_fn,'r').read().splitlines()
        total_num = 12
        count = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for q, gt, pred, imid, flag_p in zip(que_fn, ans_fn, val_pred, img_fn, flag_fn):
            imid_full=imid.zfill(total_num)
            img_name = self.img_dir + imid_full + ".jpg"
            if flag_p==str('1'):
                color_choice = 'green'
            else:
                color_choice = 'red'
                
            if count%1000==0:
                img = cv2.imread(img_name)
                plt.figure()
                plt.imshow(img)
                #plt.title(str(q) + ': ' + str(pred))
                plt.text(0,img.shape[0]+20,str(q) + ': ' + str(pred), color=color_choice)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            count += 1
        return                

    def eval_mlp(self):

        #step-0: load the graph:
        sess=tf.Session()
        graph=tf.get_default_graph()

        numeric_ans_local = joblib.load("../model/numeric_ans.pkl")              

        # get prediction resutls:
        prediction_answer_store = []

        # get the val and test data:
        ans_fn = open(self.val_ans_fn,'r').read().splitlines()
        que_fn = open(self.val_que_fn,'r').read().splitlines()
        img_fn = open(self.val_imgid_fn,'r').read().splitlines()

        k=0       

        with sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('../model/model.ckpt.meta')
            print('loaded meta graph')
            saver.restore(sess,tf.train.latest_checkpoint('../model/'))
            print("Model restored")

            saver.restore(sess, "../model/model.ckpt")
            print("Model restored.")

            x_inp = graph.get_tensor_by_name("input/x_inp:0")
            
            predictions = graph.get_tensor_by_name("val_out/val_pred:0")

            # print all tensors in checkpoint file
            chkp.print_tensors_in_checkpoint_file("../model/model.ckpt", tensor_name='', all_tensors=True, all_tensor_names=True)

            for i in range(0, len(que_fn), bs_mlp):
                if (bs_mlp*k>=len(que_fn)):
                    break
                print('evaluating batch ', i)
                que_batch = que_fn[bs_mlp*k:bs_mlp*(k+1)]
                ans_batch = ans_fn[bs_mlp*k:bs_mlp*(k+1)]
                img_batch = img_fn[bs_mlp*k:bs_mlp*(k+1)]

                #x_batch, y_batch, im_batch = self.get_input_for_nn_model(que_batch, ans_batch, img_batch)
                x_batch = self.get_que_vec(que_batch)
                im_batch = self.get_img_vec(img_batch)
                x_batch = np.hstack((x_batch, im_batch))

                #feed_dict = {x_inp: x_batch, y_out: y_batch}
                feed_dict ={x_inp: x_batch}

                prediction_answer = tf.argmax(predictions, 1)
                prediction_answer_final = sess.run(prediction_answer, feed_dict)
                prediction_answer_store.extend(numeric_ans_local.inverse_transform(prediction_answer_final))
                k+=1

        score = 0.0
        # now lets write the results:
               
        write_pred = open(self.pred_fn,'w')
        write_flag = open(self.pred_flag_fn, 'w')
        right_wrong_flag = 0
        que_for_file = {}
        ans_for_file = {}
        imid_for_file = []

        for q, gt, pred, imid in zip(que_fn, ans_fn, prediction_answer_store, img_fn):
            atleast_3 = 0
            for gt_ans in gt.split(';'):
                if pred == gt_ans:
                    atleast_3 += 1
            if atleast_3 >= 3:
                score += 1
                right_wrong_flag = 1
            elif (atleast_3>0) and (atleast_3 < 3):
                score += float(atleast_3)/3
                right_wrong_flag = 0
            else:
                right_wrong_flag = 0
            # now i need to write only the pred answer and the right_or_wrong_flag
            write_pred.write(str(pred) + '\n')
            write_flag.write(str(right_wrong_flag)+'\n')             

        write_pred.close()
        write_flag.close()
        print("Average Accuracy : " + str(score/len(que_fn)))           

        # call visualization routine:
        self.visual_results()

        return

    # check this
    def variable_summaries(self, var):
        # Attach a lot of summaries to a Tensor (for TensorBoard visualization)
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)

    def mlp_n(self):
        print('Building MLP model')

        with tf.name_scope('input'):
            x_inp = tf.placeholder("float", [None, imdim_mlp+txtdim_mlp], name="x_inp")
            y_out = tf.placeholder("float", [None, self.num_cat], name= "y_out") # should be 1000

        # create a scope and summary for input image (take the corresponding image id and read it from raw_images folder

        with tf.name_scope('weights'):
            weights = {
                'h1': tf.Variable(tf.random_normal([imdim_mlp+txtdim_mlp, hu_mlp]), name="h1"),
                'h2': tf.Variable(tf.random_normal([hu_mlp, hu_mlp]), name="h2"),
                'out': tf.Variable(tf.random_normal([hu_mlp, self.num_cat]), name="out"),
                }
            self.variable_summaries(weights['h1'])
            self.variable_summaries(weights['h2'])
            self.variable_summaries(weights['out'])

        with tf.name_scope('bias'):
            biases = {
                'b1': tf.Variable(tf.random_normal([hu_mlp]), name="b1"),
                'b2': tf.Variable(tf.random_normal([hu_mlp]), name="b2"),
                'out': tf.Variable(tf.random_normal([self.num_cat]), name="out"),
                }
            self.variable_summaries(biases['b1'])
            self.variable_summaries(biases['b2'])
            self.variable_summaries(biases['out'])
        
        #keep_prob = tf.placeholder("float") 

        #Hidden Layer 1
        with tf.name_scope('layer1'):
             with tf.name_scope('Wx_plus_b'):
                layer_1 = tf.add(tf.matmul(x_inp, weights['h1']), biases['b1'])
                tf.summary.histogram('pre_activations_layer1', layer_1)
                layer_1 = tf.nn.tanh(layer_1)
                tf.summary.histogram('activations_layer1', layer_1)
                layer_1 = tf.nn.dropout(layer_1, drop_mlp)
                tf.summary.histogram('dropout_layer1', layer_1)

        #Hidden Layer 2
        with tf.name_scope('layer2'):
             with tf.name_scope('Wx_plus_b'):
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
                tf.summary.histogram('pre_activations_layer2', layer_2)
                layer_2 = tf.nn.tanh(layer_2)
                tf.summary.histogram('activations_layer2', layer_2)
                layer_2 = tf.nn.dropout(layer_2, drop_mlp)
                tf.summary.histogram('dropout_layer2', layer_2)
                
        with tf.name_scope('out_layer'):
            #out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            out_layer = tf.add(tf.matmul(layer_2, weights['out']),biases['out'], name="out_f")
        tf.summary.histogram('out_layer', out_layer)
        #out_layer = tf.nn.softmax(out_layer)
        cr_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=y_out, name="cross_entropy")
        tf.summary.histogram('cross_entropy', cr_entropy)

        with tf.name_scope('cost'):
            cost = tf.reduce_mean(cr_entropy, name='cost')
        tf.summary.histogram('cost', cost)

        with tf.name_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)

        sess = tf.Session()

        # Merge all the summaries and write them out to
        # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(self.log_dir + '/test')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        print("Beginnning training now")
        with sess:
            sess.run(tf.global_variables_initializer())
            num_iter = epoch_mlp*len(self.train_que_final)
            num_iter/= bs_mlp
            print('Total number of iterations is, ', int(num_iter))
            k=0
            # begin iteration and training:
            for i in range(int(num_iter)):
                que_batch = self.train_que_final[bs_mlp*k:bs_mlp*(k+1)]
                ans_batch = self.train_ans_final[bs_mlp*k:bs_mlp*(k+1)]
                img_batch = self.train_img_final[bs_mlp*k:bs_mlp*(k+1)]

                x_batch, y_batch, im_batch = self.get_input_for_nn_model(que_batch, ans_batch, img_batch)
                x_batch = np.hstack((x_batch, im_batch))

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, this_loss, summary = sess.run([optimizer, cost, merged],
                                                 feed_dict={
                                                     x_inp: x_batch,
                                                     y_out: y_batch},
                                                 options=run_options,
                                                 run_metadata=run_metadata)
                #print the loss every 50th iteration :
                if i%50 == 0:
                    print("Iteration ", i, " : Loss = ", this_loss)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    train_writer.flush()
                    print('Adding run metadata for', i)
                k+=1
                if (bs_mlp*k)>=len(self.train_que_final):
                    k = 0
                    
            # Save the variables to disk.
            save_path = saver.save(sess, "../model/model.ckpt")
            print("Model saved in path: %s" % save_path)

            train_writer.close()
            test_writer.close()
            
        sess.close()
        return

    def eval_lstm(self):
        #step-0: load the graph:
        sess=tf.Session()    
        #First load meta graph and restore weights
        saver = tf.train.import_meta_graph('../model/model.ckpt.meta')
        print('loaded meta graph')
        saver.restore(sess,tf.train.latest_checkpoint('../model/'))
        graph = tf.get_default_graph()
        print("Model restored")

        numeric_ans_local = joblib.load("../model/numeric_ans.pkl")              

        # get prediction resutls:
        prediction_answer_store = []

        # get the val and test data:
        ans_fn = open(self.val_ans_fn,'r').read().splitlines()
        que_fn = open(self.val_que_fn,'r').read().splitlines()
        img_fn = open(self.val_imgid_fn,'r').read().splitlines()

        que_fn, ans_fn, img_fn = (list(t) for t in zip(*sorted(zip(que_fn, ans_fn, img_fn))))

        k=0

        x_in = graph.get_tensor_by_name("input/x_in:0")
        im_inp = graph.get_tensor_by_name("input/im_inp:0")
        #y_out = graph.get_tensor_by_name("input/y_out:0")

        #predictions = graph.get_tensor_by_name("out_layer/out_f:0")
        predictions = graph.get_tensor_by_name("val_out/val_pred:0")
        

        with sess:

            for i in range(0, len(que_fn), bs_mlp):
                if (bs_mlp*k>=len(que_fn)):
                    break
                print('evaluating batch ', i)
                que_batch = que_fn[bs_mlp*k:bs_mlp*(k+1)]
                ans_batch = ans_fn[bs_mlp*k:bs_mlp*(k+1)]
                img_batch = img_fn[bs_mlp*k:bs_mlp*(k+1)]

                num_cells = len(self.get_token(que_batch[-1]))
                x_batch = self.get_que_vec_lstm(que_batch, max_que_len)
                im_batch = self.get_img_vec(img_batch)

                feed_dict ={x_in: x_batch, im_inp: im_batch}
                
                prediction_answer = tf.argmax(predictions,1)
                prediction_answer_final = sess.run(prediction_answer, feed_dict)
                prediction_answer_store.extend(numeric_ans_local.inverse_transform(prediction_answer_final))
                k+=1

        score = 0.0
        # now lets write the results:
               
        write_pred = open(self.pred_fn,'w')
        write_flag = open(self.pred_flag_fn, 'w')
        right_wrong_flag = 0
        que_for_file = {}
        ans_for_file = {}
        imid_for_file = []

        for q, gt, pred, imid in zip(que_fn, ans_fn, prediction_answer_store, img_fn):
            atleast_3 = 0
            for gt_ans in gt.split(';'):
                if pred == gt_ans:
                    atleast_3 += 1
            if atleast_3 >= 3:
                score += 1
                right_wrong_flag = 1
            elif (atleast_3 >0 and atleast_3 < 3):
                score += float(atleast_3)/3
                right_wrong_flag = 0
            else:
                right_wrong_flag = 0
            # now i need to write only the pred answer and the right_or_wrong_flag
            write_pred.write(str(pred) + '\n')
            write_flag.write(str(right_wrong_flag)+'\n')

        write_pred.close()
        write_flag.close()
        print("Average Accuracy : " + str(score/len(que_fn)))           

        # call visualization routine:
        self.visual_results()

        return
                
    def lstm_m(self):
        
        # self.train_que_final, self.train_ans_final, self.train_img_final
        self.train_quelen_final = open(self.train_quelen_fn, 'r').read().splitlines()
        # sort questions by their length
        que_train, quelen_train, ans_train, imgid_train = (list(t) for t in zip(*sorted(zip(self.train_que_final, self.train_quelen_final, self.train_ans_final, self.train_img_final))))

        print('Building LSTM model')

        # first let's build a LSTM:        
        with tf.name_scope('input'):
            x_in = tf.placeholder("float", [None, max_que_len, txtdim_mlp], name="x_in")
            x_inp = tf.unstack(x_in, max_que_len, 1,name="x_inp")
            #num_cells = tf.placeholder(tf.int32, name="num_cells")
            #x_inp = tf.unstack(x_in, num_cells[0], 1, name="x_inp")
            #x_inp = tf.placeholder("float", [max_que_len, txtdim_mlp], name="x_in")
            im_inp = tf.placeholder("float", [None, imdim_mlp], name="im_inp")
            y_out = tf.placeholder("float", [None, self.num_cat], name= "y_out") # should be 1000

        with tf.name_scope('weights'):
            weights = {
                'h1': tf.Variable(tf.random_normal([hu_lstm+imdim_mlp, hu_mlp_lstm]), name="h1"),
                'h2': tf.Variable(tf.random_normal([hu_mlp_lstm, hu_mlp_lstm]), name="h2"),
                'h3': tf.Variable(tf.random_normal([hu_mlp_lstm, hu_mlp_lstm]), name = "h3"),
                'out': tf.Variable(tf.random_normal([hu_mlp_lstm, self.num_cat]), name="out"),
                }
            self.variable_summaries(weights['h1'])
            self.variable_summaries(weights['h2'])
            self.variable_summaries(weights['h3'])
            self.variable_summaries(weights['out'])

        with tf.name_scope('bias'):
            biases = {
                'b1': tf.Variable(tf.random_normal([hu_mlp]), name="b1"),
                'b2': tf.Variable(tf.random_normal([hu_mlp]), name="b2"),
                'b3': tf.Variable(tf.random_normal([hu_mlp]), name="b3"),
                'out': tf.Variable(tf.random_normal([self.num_cat]), name="out"),
                }
            self.variable_summaries(biases['b1'])
            self.variable_summaries(biases['b2'])
            self.variable_summaries(biases['b3'])
            self.variable_summaries(biases['out'])

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hu_lstm, forget_bias=1.0)

        # Get lstm cell output
        #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x_inp, sequence_length=max_que_len)
        outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x_inp, dtype=tf.float32)

        # create a scope and summary for input image (take the corresponding image id and read it from raw_images folder
        
        input_layer = tf.concat([outputs[-1], im_inp], axis= 1) #512+4096 = 4608
        #Hidden Layer 1
        with tf.name_scope('layer1'):
             with tf.name_scope('Wx_plus_b'):
                layer_1 = tf.add(tf.matmul(input_layer, weights['h1']), biases['b1'])
                tf.summary.histogram('pre_activations_layer1', layer_1)
                layer_1 = tf.nn.tanh(layer_1)
                tf.summary.histogram('activations_layer1', layer_1)
                layer_1 = tf.nn.dropout(layer_1, drop_mlp)
                tf.summary.histogram('dropout_layer1', layer_1)

        #Hidden Layer 2
        with tf.name_scope('layer2'):
             with tf.name_scope('Wx_plus_b'):
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
                tf.summary.histogram('pre_activations_layer2', layer_2)
                layer_2 = tf.nn.tanh(layer_2)
                tf.summary.histogram('activations_layer2', layer_2)
                layer_2 = tf.nn.dropout(layer_2, drop_mlp)
                tf.summary.histogram('dropout_layer2', layer_2)

        #Hidden Layer 3
        with tf.name_scope('layer3'):
             with tf.name_scope('Wx_plus_b'):
                layer_3 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
                tf.summary.histogram('pre_activations_layer3', layer_3)
                layer_3 = tf.nn.tanh(layer_3)
                tf.summary.histogram('activations_layer3', layer_3)
                layer_3 = tf.nn.dropout(layer_3, drop_mlp)
                tf.summary.histogram('dropout_layer3', layer_3)
                
        with tf.name_scope('out_layer'):
            #out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            out_layer = tf.add(tf.matmul(layer_3, weights['out']),biases['out'], name="out_f")
        tf.summary.histogram('out_layer', out_layer)
        #out_layer = tf.nn.softmax(out_layer)
        cr_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=y_out, name="cross_entropy")
        tf.summary.histogram('cross_entropy', cr_entropy)

        with tf.name_scope('cost'):
            cost = tf.reduce_mean(cr_entropy, name='cost')
        tf.summary.histogram('cost', cost)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

        sess = tf.Session()

        # Merge all the summaries and write them out to
        # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(self.log_dir + '/test')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        print("Beginnning training now")
        # remember tf.InteractiveSession() if session does not work !!
        with sess:
            sess.run(tf.global_variables_initializer())
            num_iter = epoch_mlp*len(self.train_que_final)
            num_iter/= bs_mlp
            print('Total number of iterations is, ', int(num_iter))
            k=0
            # begin iteration and training:
            for i in range(int(num_iter)):
                que_batch = que_train[bs_mlp*k:bs_mlp*(k+1)]
                ans_batch = ans_train[bs_mlp*k:bs_mlp*(k+1)]
                img_batch = imgid_train[bs_mlp*k:bs_mlp*(k+1)]

                # check
                num_cells = len(self.get_token(que_batch[-1]))
                x_batch = self.get_que_vec_lstm(que_batch, max_que_len)
                y_batch = self.get_ans_vec(ans_batch)
                im_batch = self.get_img_vec(img_batch)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, this_loss, summary = sess.run([optimizer, cost, merged],
                                                 feed_dict={
                                                     x_in: x_batch,
                                                     im_inp: im_batch,
                                                     y_out: y_batch},
                                                 options=run_options,
                                                 run_metadata=run_metadata)
                #print the loss every 50th iteration :
                if i%50 == 0:
                    print("Iteration ", i, " : Loss = ", this_loss)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    train_writer.flush()
                    print('Adding run metadata for', i)
                k+=1
                if (bs_mlp*k)>=len(self.train_que_final):
                    k = 0
                    
            # Save the variables to disk.
            save_path = saver.save(sess, "../model/model.ckpt")
            print("Model saved in path: %s" % save_path)

            train_writer.close()
            test_writer.close()
            
        sess.close()
        # now evaluate the trained model :
        self.eval_lstm()
        pass

    def main(self, choice):
        options = {'mlp_m':self.mlp_n,
                   'eval_mlp': self.eval_mlp,
                   'vis_m': self.visual_results,
                   'lstm_m':self.lstm_m,
                   'eval_lstm': self.eval_lstm,
            }
        #step-0: All preprocessing routines to process the text and image data
        if not(choice=='vis_m'):
            self.load_GloVe_embed()
        '''if not((choice=='eval_mlp') or (choice=='vis_m')):
            self.get_data_in_format()'''

        self.get_most_frequent_set()

        if not((choice=='eval_mlp') or (choice=='vis_m') or (choice=='eval_lstm')):
            # convert the features to numeric:
            self.numeric_ans.fit(self.train_ans_final)
            self.num_cat = len(list(self.numeric_ans.classes_))
            joblib.dump(self.numeric_ans, "../model/numeric_ans.pkl")

        #step-1: Get the extracted image features
        print("Extracting Image features")
        vgg_features = scipy.io.loadmat(self.vgg_model)
        self.img_features = vgg_features['feats']

        # now also get the mapping of vgg image id vs coco image id
        self.get_vgg_coco_map()
        
        #step-2: Build the model and train it
        options[choice]()       
        return

if __name__ == "__main__":

    # creating an instance of the vqa class
    vq_cl = VQA_Class()
    # Call to the main function
    tic = time.time()
    print('Choices are : \n 1. MLP Baseline train (mlp_m), \n 2. MLP Baseline eval (eval_mlp) \n 3. Visualize results (vis_m) \n 4. LSTM (lstm_m), \n 5. Evaluate LSTM(eval_lstm) and \n 6. MCB(mcb_m)')
    choice_v = input("Enter the choice: ")
    vq_cl.main(choice_v)
    toc = time.time() - tic
    print("Running time: " + str(toc))
