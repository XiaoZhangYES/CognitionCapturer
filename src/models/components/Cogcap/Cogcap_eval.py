import random
from torch import cosine_similarity
from torch.nn import functional as F
import torch

class Top_K_Accuracy():
    def __init__(self, logit_scale=None, k=None, modality=None, func=None):
        self.logit_scale = logit_scale
        self.k = k
        self.modality = modality

        assert func is not None, "func param can not be None!"
        self.func = getattr(self, func)

    def _visual_sim(self,
                    EEG_img_features,
                    EEG_text_features,
                    EEG_depth_features,
                    img_select_features,
                    text_select_features,
                    depth_select_features):
        """
        已知：sim在面对正确的类的时候就是好
        通过这个函数来看相似度的关系
        目前是选出最好的sim来作为评判标准
        选出来平均的sim可否? 尝试top5 平均/ top 200平均
        平均效果也不好

        在他们的top5中，分别看前面几个，通过差异度来确定最终选择？
        """
        img_sim = cosine_similarity(EEG_img_features, img_select_features, dim=1)
        text_sim = cosine_similarity(EEG_text_features, text_select_features, dim=1)
        depth_sim = cosine_similarity(EEG_depth_features, depth_select_features, dim=1)

        return img_sim, text_sim, depth_sim

    def calculate_single_modality(self,
                                  EEG_features,
                                  targets,
                                  img_features_all=None,
                                  text_features_all=None,
                                  depth_features_all=None,
                                  aug_img_features_all=None,
                                  modality_features=None,
                                  ):
        """
        CAUTION: When batch_size is smaller than k, this func will calculate using all batch's classes

        evaluate_model_classification used text_features
        Select k classes from label, Then calculate top-1 accuracy and top-5 accuracy in the classes

        :param EEG_features: original EEG_features, dim:(batch_size, dim)
        :param targets: The index classes, dim:(batchsize, )
        :return: A tensor of losses between model predictions and targets.
        """
        correct = 0
        total = 0
        top5_correct_count = 0

        if self.k != 0:
            # iterate over the batch for calculating
            for idx, label in enumerate(targets):
                # all_labels denote this batch's all_labels
                all_labels = set(range(targets.size(0)))

                possible_classes = list(all_labels - {label.item()})
                # append true class in the last
                if len(possible_classes) >= self.k - 1:
                    selected_classes = random.sample(possible_classes, self.k - 1) + [label.item()]
                else:
                    selected_classes = possible_classes + [label.item()]
                    raise RuntimeError("watch out!")


                ### select Modality ###
                if self.modality is not None and modality_features is not None:
                    raise NotImplementedError(f"Can only use one feature")
                if self.modality is not None:
                    if self.modality == "image":
                        selected_features = img_features_all[selected_classes].to(EEG_features.device)
                    elif self.modality == "text":
                        selected_features = text_features_all[selected_classes].to(EEG_features.device)
                    elif self.modality == "depth":
                        selected_features = depth_features_all[selected_classes].to(EEG_features.device)
                    else:
                        raise NotImplementedError(f"please pass modality into __init__, now modality: {self.modality}")
                elif modality_features is not None:
                    selected_features = modality_features[selected_classes].to(EEG_features.device)
                else:
                    raise NotImplementedError(f"Must assign a usage")
                ### select Modality ###

                ### logits cal by detecting img_features IMPORTANT ###
                logits = self.logit_scale * EEG_features[idx] @ selected_features.T
                ### logits cal by detecting img_features IMPORTANT ###

                # torch.argmax(logits): get logits's max value's index
                # indices are top5 indexs of modality_features
                # _, top5_indices_img = torch.topk(logits_img, 5, largest=True)
                # _, top5_indices_text = torch.topk(logits_text, 5, largest=True)
                # _, top5_indices_depth = torch.topk(logits_depth, 5, largest=True)
                # top5_indices = torch.stack([top5_indices_img, top5_indices_text, top5_indices_depth], dim=0)
                # row_sets = [set(row.tolist()) for row in top5_indices]  # sets 没有顺序
                # common_values = set.intersection(*row_sets)
                # if common_values_tensor.numel() == 1:  # 都有视为正确
                #     if label.item() == selected_classes[common_values_tensor.item()]:
                #         correct += 1

                predicted_label = selected_classes[torch.argmax(logits).item()]
                if predicted_label == label.item():
                    correct += 1
                _, top5_indices = torch.topk(logits, 5, largest=True)
                if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                    top5_correct_count += 1
                total += 1

            ### here getted eeg

        else:
            # k = 0 not using top k for calculation
            logits_img = self.logit_scale * EEG_features @ img_features_all.to(EEG_features.device).T
            predicted = torch.argmax(logits_img, dim=1)
            batch_size = predicted.shape[0]
            total += batch_size
            correct += (predicted == targets).sum().item()


        accuracy = correct / total
        top5_acc = top5_correct_count / total
        return accuracy, top5_acc


    def calculate_allmodality(self,
                              EEG_img_features=None,
                              EEG_text_features=None,
                              EEG_depth_features=None,
                              img_features_all=None,
                              text_features_all=None,
                              depth_features_all=None,
                              modality_index=None,
                              targets=None,
                              seperate_calculate=False
                              ):
        """
        eval: separately cal modality, then see upper bound
        """

        if seperate_calculate is True:
            # index: 0: img, 1: text, 2: depth, 3: all todo add edge mod 24/12/6
            correct = [0, 0, 0, 0]
            total = [0, 0, 0, 0]
            top5_correct_count = [0, 0, 0, 0]
        else:
            correct = 0
            total = 0
            top5_correct_count = 0

        if self.k != 0:
            # targets: (batch, ) # denotes every eeg's label
            # iterate over the label
            for idx, label in enumerate(targets):
                if modality_index is not None:
                    modality_index_single = modality_index[idx]

                # all_labels denote this batch's all_labels
                all_labels = set(range(targets.size(0)))

                possible_classes = list(all_labels - {label.item()})
                # append true class in the last
                if len(possible_classes) >= self.k - 1:
                    # select k-1 classes and true label add
                    selected_classes = random.sample(possible_classes, self.k - 1) + [label.item()]
                else:
                    selected_classes = possible_classes + [label.item()]
                    raise RuntimeError("watch out!")

                img_select_features = img_features_all[selected_classes].to(EEG_img_features.device)
                text_select_features = text_features_all[selected_classes].to(EEG_text_features.device)
                depth_select_features = depth_features_all[selected_classes].to(EEG_depth_features.device)

                ### logits cal by detecting img_features IMPORTANT ###
                sim = self._visual_sim(
                    EEG_img_features[idx],
                    EEG_text_features[idx],
                    EEG_depth_features[idx],
                    img_select_features,
                    text_select_features,
                    depth_select_features
                )
                ### logits cal by detecting img_features IMPORTANT ###

                # indices are top5 indexs of modality_features
                _, top5_indices_img = torch.topk(sim[0], 5, largest=True)
                _, top5_indices_text = torch.topk(sim[1], 5, largest=True)
                _, top5_indices_depth = torch.topk(sim[2], 5, largest=True)
                top5_indices = torch.stack([top5_indices_img, top5_indices_text, top5_indices_depth], dim = 0)


                row_sets = [set(row.tolist()) for row in top5_indices] # sets 没有顺序
                common_values = set.intersection(*row_sets)
                common_values_tensor = torch.tensor(list(common_values), dtype=top5_indices.dtype)

                if seperate_calculate is True:
                    # 3 modality 分别进行计算
                    for idx in range(3):
                        ### see top1 acc ###
                        if common_values_tensor.numel() == 1 and label.item() == selected_classes[common_values_tensor.item()]:
                            correct[idx] += 1
                        elif label.item() == selected_classes[top5_indices[idx].tolist()[0]]:
                            correct[idx] += 1
                        ### see top1 acc ###

                        ### see top5 acc ###
                        if label.item() in [selected_classes[i] for i in top5_indices[idx].tolist()]:
                            correct_note = True  # if top1 are true, top 5 are definitely true
                            top5_correct_count[idx] += 1
                        ### see top5 acc ###

                        total[idx] += 1
                    ### 专门对all mod的情况进行处理 ###
                    for idx in range(3):  # 3 modality

                        correct_note = False
                        ### see top1 acc ###
                        if common_values_tensor.numel() == 1:
                            if label.item() == selected_classes[common_values_tensor.item()]:
                                correct[3] += 1
                        elif label.item() == selected_classes[top5_indices[idx].tolist()[0]]:
                            correct[3] += 1
                        ### see top1 acc ###

                        ### see top5 acc ###
                        if label.item() in [selected_classes[i] for i in top5_indices[idx].tolist()]:
                            correct_note = True  # if top1 are true, top 5 are definitely true
                            top5_correct_count[3] += 1
                        ### see top5 acc ###

                        if correct_note is True:  # when any modality is correct, break
                            total[3] += 1
                            break
                        elif idx == 2:
                            total[3] += 1
                            break

                else:
                    if modality_index is not None:
                        correct_note = False
                        ### see top1 acc ###
                        if common_values_tensor.numel() == 1: #都有视为正确
                            if label.item() == selected_classes[common_values_tensor.item()]:
                                correct += 1
                        if label.item() == selected_classes[top5_indices[modality_index_single.item()].tolist()[0]]:
                            correct += 1
                        ### see top1 acc ###

                        ### see top5 acc ###
                        if label.item() in [selected_classes[i] for i in top5_indices[modality_index_single.item()].tolist()]:
                            correct_note = True  # if top1 are true, top 5 are definitely true
                            top5_correct_count += 1
                        ### see top5 acc ###
                        total += 1
                    else:
                        for idx in range(3):  # 3 modality
                            correct_note = False

                            ### see top1 acc ###
                            if common_values_tensor.numel() == 1:
                                if label.item() == selected_classes[common_values_tensor.item()]:
                                    correct += 1
                            elif label.item() == selected_classes[top5_indices[idx].tolist()[0]]:
                                correct += 1
                            ### see top1 acc ###

                            ### see top5 acc ###
                            if label.item() in [selected_classes[i] for i in top5_indices[idx].tolist()]:
                                correct_note = True  # if top1 are true, top 5 are definitely true
                                top5_correct_count += 1
                            ### see top5 acc ###

                            if correct_note is True:  # when any modality is correct, break
                                total += 1
                                break
                            elif idx == 2:
                                total += 1
                                break


        else:
            raise NotImplementedError("must assign K value!")


        if seperate_calculate is False:
            accuracy = correct / total
            top5_acc = top5_correct_count / total

            return accuracy, top5_acc
        else:
            acc = [0, 0, 0, 0]
            top5_acc = [0, 0, 0, 0]
            for idx in range(4):
                acc[idx] = correct[idx] / total[idx]
                top5_acc[idx] = top5_correct_count[idx] / total[idx]
            return acc, top5_acc

    def calculate(self,
                  EEG_img_features=None,
                  EEG_text_features=None,
                  EEG_depth_features=None,
                  img_features_all=None,
                  text_features_all=None,
                  depth_features_all=None,
                  modality_index=None,
                  targets=None,
                  seperate_calculate=False
                  ):
        """
        main entrance, use this for easy coding
        """
        return self.func(
            EEG_img_features,
            EEG_text_features,
            EEG_depth_features,
            img_features_all,
            text_features_all,
            depth_features_all,
            modality_index,
            targets,
            seperate_calculate
        )




if __name__ == "__main__":
    ### Get EEG correlated data ###
    EEG_features = torch.randn(1000, 1024).to("cuda:1")
    EEG_img_features = torch.randn(1000, 1024).to("cuda:1")
    EEG_depth_features = torch.randn(1000, 1024).to("cuda:1")
    EEG_text_features = torch.randn(1000, 1024).to("cuda:1")
    EEG_data = torch.randn(1000, 63, 250).to("cuda:1")
    ### Get EEG correlated data ###

    ### Ground truth data ###
    img_features_all = torch.randn(1000, 1024).to("cuda:1")
    text_features_all = torch.randn(1000, 1024).to("cuda:1")
    depth_features_all = torch.randn(1000, 1024).to("cuda:1")
    aug_img_features_all = torch.randn(1000, 1024).to("cuda:1")
    targets = torch.arange(0, 1000).to("cuda:1")  # target 代表 label 即图像的类别
    ### Ground truth data ###

    logit_scale = 10
    alpha = 0.5
    k = 200
    top5 = Top_K_Accuracy(logit_scale, k, func="calculate_allmodality")

    acc, top5 = top5.calculate(
        EEG_img_features=EEG_img_features,
        EEG_depth_features=EEG_depth_features,
        EEG_text_features=EEG_text_features,
        targets=targets,
        img_features_all=img_features_all,
        text_features_all=text_features_all,
        depth_features_all=depth_features_all,
        seperate_calculate=True,
    )
    print(acc, top5)
