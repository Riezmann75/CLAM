import torch
import numpy as np


def c_index(predicted_risks, event_times, event_indicators):
    """
    Calculate the concordance index (C-index) for survival analysis.

    Args:
        predicted_risks: torch.Tensor or np.array of predicted risk scores from model
                        Higher values = higher risk of event
        event_times: torch.Tensor or np.array of observed times (either event or censoring times)
        event_indicators: torch.Tensor or np.array of event indicators
                         (1 = event occurred, 0 = censored)

    Returns:
        float: C-index value between 0 and 1
               - 0.5 = random predictions
               - 1.0 = perfect concordance
               - 0.0 = perfect anti-concordance
    """
    # Convert to numpy if torch tensors
    if isinstance(predicted_risks, torch.Tensor):
        predicted_risks = predicted_risks.detach().cpu().numpy()
    if isinstance(event_times, torch.Tensor):
        event_times = event_times.detach().cpu().numpy()
    if isinstance(event_indicators, torch.Tensor):
        event_indicators = event_indicators.detach().cpu().numpy()

    # Flatten arrays
    predicted_risks = predicted_risks.flatten()
    event_times = event_times.flatten()
    event_indicators = event_indicators.flatten()

    n = len(predicted_risks)
    concordant = 0
    discordant = 0
    tied_risk = 0
    tied_time = 0

    # Compare all valid pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Skip pairs where both are censored
            if event_indicators[i] == 0 and event_indicators[j] == 0:
                continue

            # Determine comparable pairs
            # Case 1: Both experienced events
            if event_indicators[i] == 1 and event_indicators[j] == 1:
                if event_times[i] < event_times[j]:
                    shorter_time_i = True
                elif event_times[i] > event_times[j]:
                    shorter_time_i = False
                else:
                    tied_time += 1
                    continue

            # Case 2: One censored, one event
            elif event_indicators[i] == 1 and event_indicators[j] == 0:
                # i had event, j was censored
                if event_times[i] <= event_times[j]:
                    shorter_time_i = True
                else:
                    continue  # Not comparable

            elif event_indicators[i] == 0 and event_indicators[j] == 1:
                # i was censored, j had event
                if event_times[j] <= event_times[i]:
                    shorter_time_i = False
                else:
                    continue  # Not comparable
            else:
                continue

            # Check concordance
            if predicted_risks[i] == predicted_risks[j]:
                tied_risk += 1
            elif shorter_time_i and predicted_risks[i] > predicted_risks[j]:
                concordant += 1
            elif not shorter_time_i and predicted_risks[j] > predicted_risks[i]:
                concordant += 1
            else:
                discordant += 1

    # print(f"Concordant: {concordant}")
    # print(f"Discordant: {discordant}")
    # print(f"Tied Risk: {tied_risk}")
    # print(f"Tied Time: {tied_time}")
    # Calculate C-index
    total_comparable = concordant + discordant + tied_risk

    if total_comparable == 0:
        return 0.5  # No comparable pairs

    # C-index with tied risk handling
    c_index = (concordant + 0.5 * tied_risk) / total_comparable

    return c_index


def dynamic_c_index(model, features, event_times, event_indicators):
    """Calculate the C index by computing the risk scores at each unique event time.

    Args:
        model (_type_): _survival model with a method `predict_risk` that takes event times as input_
        event_times (_type_): _array of observed times (either event or censoring times)_
        event_indicators (_type_): _array of event indicators (1 = event occurred, 0 = censored)_
    """

    # Convert to numpy if torch tensors
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(event_times, torch.Tensor):
        event_times = event_times.detach().cpu().numpy()
    if isinstance(event_indicators, torch.Tensor):
        event_indicators = event_indicators.detach().cpu().numpy()

    event_times = event_times.flatten()
    event_indicators = event_indicators.flatten()

    n = len(event_times)
    concordant = 0
    discordant = 0
    tied_time = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Skip pairs where both are censored
            if event_indicators[i] == 0 and event_indicators[j] == 0:
                continue

            # Determine comparable pairs
            # Case 1: Both experienced events
            if event_indicators[i] == 1 and event_indicators[j] == 1:
                if event_times[i] < event_times[j]:
                    shorter_time_i = True
                elif event_times[i] > event_times[j]:
                    shorter_time_i = False
                else:
                    tied_time += 1
                    continue

            # Case 2: One censored, one event
            elif event_indicators[i] == 1 and event_indicators[j] == 0:
                # i had event, j was censored
                if event_times[i] <= event_times[j]:
                    shorter_time_i = True
                else:
                    continue  # Not comparable

            elif event_indicators[i] == 0 and event_indicators[j] == 1:
                # i was censored, j had event
                if event_times[j] <= event_times[i]:
                    shorter_time_i = False
                else:
                    continue  # Not comparable
            else:
                continue

            # Check concordance
            if shorter_time_i:
                tensor_i = torch.tensor(features[[i]], dtype=torch.float32)
                tensor_j = torch.tensor(features[[j]], dtype=torch.float32)
                time_i = torch.tensor(event_times[[i]], dtype=torch.float32)
                risk_i = model(tensor_i, time_i)
                risk_j = model(tensor_j, time_i)
                if risk_i > risk_j:
                    concordant += 1
                elif risk_i < risk_j:
                    discordant += 1
                else:
                    tied_risk += 1
            else:
                tensor_i = torch.tensor(features[[i]], dtype=torch.float32)
                tensor_j = torch.tensor(features[[j]], dtype=torch.float32)
                time_j = torch.tensor(event_times[[j]], dtype=torch.float32)
                risk_i = model(tensor_i, time_j)
                risk_j = model(tensor_j, time_j)
                if risk_j > risk_i:
                    concordant += 1
                elif risk_j < risk_i:
                    discordant += 1
                else:
                    tied_risk += 1

    print(f"Concordant: {concordant}")
    print(f"Discordant: {discordant}")
    print(f"Tied Time: {tied_time}")
    print(f"Tied Risk: {tied_risk}")
    # Calculate C-index
    total_comparable = concordant + discordant + tied_risk

    if total_comparable == 0:
        return 0.5  # No comparable pairs

    # C-index with tied risk handling
    c_index = (concordant + 0.5 * tied_risk) / total_comparable

    return c_index
