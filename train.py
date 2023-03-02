from utils.setting import *
from utils.evaluate import *
import torch
import numpy as np
from test import test

def train(model, save_path, config, data_config, train_loader, val_loader, test_loader, viz=None, viz_test=False):
    device = get_device()
    model.to(device)
    if model.name == 'AGVAE':
        optimizer = torch.optim.Adam([i for i in model.encoder.parameters()]+[j for j in model.decoder.parameters()], lr=config['lr'], weight_decay=config['decay'])
        discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['decay'])
    early_stop_win = config['early_stop_win']
    min_val_loss = 1e8

    for epoch in range(config['epoch']):
        model.train()
        epoch_train_loss = 0
        for batch_data in train_loader:
            new_batch_data = model(batch_data)
            train_result_graph_list = new_batch_data.to_data_list()
            if model.name == 'AGVAE':
                for discriminator_epoch in range(5):
                    discriminator_loss = 0
                    model.discriminator.train()
                    model.discriminator.zero_grad()
                    new_batch_data_list = new_batch_data.to_data_list()
                    for graph in new_batch_data_list:
                        discriminator_loss += model.discriminator_loss(graph.z)
                    discriminator_loss.backward()
                    discriminator_optimizer.step()
            num_sensors, window_size = train_result_graph_list[0].x.shape
            batch_loss = torch.nn.functional.mse_loss(new_batch_data.x, new_batch_data.x_) * window_size * config['alpha']
            if model.name == 'STAGE':
                for graph in train_result_graph_list:
                    delay_weight = torch.arange(1, window_size+1).unsqueeze(0).repeat_interleave(num_sensors, 0).to(device)
                    delay_weight = (torch.ones_like(delay_weight) * window_size - delay_weight)/(delay_weight**2)
                    weighted_pred_error = delay_weight * torch.nn.functional.mse_loss(graph.pred, graph.pred_, reduction='none')
                    weighted_pred_error = weighted_pred_error.sum()/num_sensors
                    batch_loss += weighted_pred_error * (1 - config['alpha'])
            if model.name == 'GVAD' or model.name == 'AGVAE':
                batch_loss += (new_batch_data.kl_loss.sum()/(len(batch_data.y) * batch_data.x.shape[1] * batch_data.x.shape[0]))
            epoch_train_loss += batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        avg_train_loss = epoch_train_loss/(len(train_loader) * data_config['batch_size'])
        print(f'In epoch {epoch}, the loss is {avg_train_loss}')
        updates = 'append' if epoch != 0 else None
        if viz is not None:
            viz.line(X=np.array([epoch]), Y=np.array([avg_train_loss.cpu().detach().numpy()]), win='Train_Val_Loss',
                                                         name='train_loss', update=updates, opts={'showlegend': True,
                                                                                                  'title': 'Train Val Loss',
                                                                                                  'xlabel': 'epoch',
                                                                                                  'ylabel': 'loss'})

        describe = config['comment']
        torch.save(model.state_dict(), f'./pretrained/{data_config["use_dataset"]}/{describe}/{data_config["use_dataset"]}_model_@_{epoch}.pt')

        if val_loader is not None:
            stop_improve_count = 0
            val_loss, val_result, val_recon = test(model, val_loader, config, data_config)
            viz.line(X=np.array([epoch]), Y=np.array([val_loss.cpu().detach().numpy()]), win='Train_Val_Loss',
                                                         name='val_loss', update='append', opts={'showlegend': True,
                                                                                                 'title': 'Train Val Loss',
                                                                                                 'xlabel': 'epoch',
                                                                                                 'ylabel': 'loss'})
            print(f'In epoch {epoch}, the val loss is {val_loss}')

            if val_loss < min_val_loss:
                now_save_path = save_path
                torch.save(model.state_dict(), now_save_path)

                min_val_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1
            if stop_improve_count >= early_stop_win:
                break

        else:
            if epoch_train_loss < min_val_loss:
                now_save_path = save_path
                torch.save(model.state_dict(), now_save_path)
                min_val_loss = epoch_train_loss

        if viz_test:
            with torch.no_grad():
                model.eval()
                test_label = []
                test_error = []
                for batch_data_test in test_loader:
                    batch_data_test_new = model(batch_data_test)
                    graph_list_test = batch_data_test_new.to_data_list()
                    for graph in graph_list_test:
                        x_raw, x_reconstruct = graph.x, graph.x_  # x: num_features * window_size
                        num_features, window_size = graph.x.shape
                        span_error = []
                        for i in range(len(x_raw)):
                            # feature_error = dtw_reconstruction_error(x_raw[i, :], x_reconstruct[i, :])
                            # feature_error = torch.nn.functional.mse_loss(x_raw[i, :], x_reconstruct[i, :]).cpu().detach().numpy()
                            feature_error = torch.nn.functional.mse_loss(x_raw[i, :], x_reconstruct[i, :]).cpu().detach().numpy()
                            span_error.append(feature_error)
                        test_error.append(span_error)
                        test_label.append(graph.y.item())

                val_error = []
                for batch_data_val in val_loader:
                    batch_data_val_new = model(batch_data_val)
                    graph_list_val = batch_data_val_new.to_data_list()
                    for graph in graph_list_val:
                        x_raw, x_reconstruct = graph.x, graph.x_
                        span_error_val = []
                        for i in range(len(x_raw)):
                            feature_error = torch.nn.functional.mse_loss(x_raw[i, :], x_reconstruct[i, :]).cpu().detach().numpy()
                            span_error_val.append(feature_error)
                        val_error.append(span_error_val)

                viz.line(Y=val_error, X=np.arange((len(val_error))), win='val_feature_error', opts={'showlegend': True,
                                                                                                  'title': f'Val_Channel_Error_at_{epoch}',
                                                                                                  'xlabel': 'Time',
                                                                                                  'ylabel': 'Anomaly_Score'})

                viz.line(Y=test_error, X=np.arange((len(test_error))), win='feature_error',  opts={'showlegend': True,
                                                                   'title': f'Test_Channel_Error_at_{epoch}',
                                                                   'xlabel': 'Time',
                                                                   'ylabel': 'Anomaly_Score'})


                if epoch ==0:
                    viz.line(Y=test_label, X=np.arange(len(test_label)), name='label', win='label', opts={'showlegend': True,
                                                                       'title': f'Test_Label_at_{epoch}',
                                                                       'xlabel': 'Time',
                                                                       'ylabel': 'Label'})
                normalized_test_error = normalize_error(test_error, val_error, method='std', dataset='swat', beta=2.5)

                viz.line(Y= normalized_test_error, X=np.arange(len(normalized_test_error)), win=f'normalized_channel_error_at_{epoch}', opts={'showlegend': True,
                                                                   'title': f'Normalized_Channel_Error_at_{epoch}',
                                                                   'xlabel': 'Time',
                                                                   'ylabel': 'Anomaly_Scores'})