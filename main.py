import mne
import scipy.signal
from numpy import linalg
from scipy import signal
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Lista dei file .gdf che vuoi combinare
file_paths = [
r'C:\Users\Fast9\PycharmProjects\PythonProject\data\BCICIV_2b_gdf\B0301T.gdf',
r'C:\Users\Fast9\PycharmProjects\PythonProject\data\BCICIV_2b_gdf\B0302T.gdf',
r'C:\Users\Fast9\PycharmProjects\PythonProject\data\BCICIV_2b_gdf\B0303T.gdf'
]
#funziona bene su soggetto 1,4,6

# Carica il primo file come base
raw_combined = mne.io.read_raw_gdf(file_paths[0], preload=True)
# Aggiungi gli altri file
for path in file_paths[1:]:
    raw_temp = mne.io.read_raw_gdf(path, preload=True)
    raw_combined.append(raw_temp, preload=True)
# Ora raw_combined contiene tutti i dati EEG uniti
raw_combined._data *= 1e6  # da volt a microvolt
# Seleziona solo i canali EEG (C3 Cz C4)
raw_combined.pick_channels(['EEG:C3',"EEG:Cz", 'EEG:C4'])

# Esamina i dati dopo la selezione
print(f"Shape dei dati EEG dopo la selezione dei canali C3, Cz e C4: {raw_combined.get_data().shape}")
# Calcola la matrice di covarianza per i dati selezionati
cov_matrix = np.cov(raw_combined.get_data())
print(f"Forma della matrice di covarianza : {cov_matrix.shape}")
sample_rate = raw_combined.info['sfreq']
EEG = raw_combined.get_data()
nchannels, ltime = EEG.shape


# Estrai informazioni sugli eventi
events, event_ids = mne.events_from_annotations(raw_combined)
# Ottieni i numeri corrispondenti ai codici 769 e 770
event_code_769 = event_ids.get('769')  # Codice 769
event_code_770 = event_ids.get('770')  # Codice 770
classes_of_interest=[event_code_769,event_code_770]
events = events[np.isin(events[:, 2], classes_of_interest)]
event_onsets = events[:, 0]
event_codes = events[:, 2]
event_ids = {key: value for key, value in event_ids.items() if value in classes_of_interest}
channel_names = raw_combined.ch_names



cl_lab = list(event_ids.keys())
cl1, cl2 = cl_lab[-2:]

nclasses = len(cl_lab)
nevents = len(event_onsets)

print('Shape of EEG:', EEG.shape)
print('Sample rate:', sample_rate)
print('Number of channels:', nchannels)
print('Channel names:', channel_names)
print('Number of events:', len(event_onsets))
print('Event codes:', np.unique(event_codes))
print('Class labels:', cl_lab)
print('Number of classes:', nclasses)

trials = {}

# Finestra temporale: da 0.5 a 2.5 secondi
start, end = int(0.5 * sample_rate), int(2.5 * sample_rate)
win = np.arange(start, end)
nsamples = end - start

# Loop over the classes
for cl, code in zip(cl_lab, np.unique(event_codes)):

    # Extract the onsets for the class
    cl_onsets = event_onsets[event_codes == code]

    # Allocate memory for the trials
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))

    # Extract each trial
    for i, onset in enumerate(cl_onsets):
        trials[cl][:, :, i] = EEG[:, win + onset]

labels = np.concatenate([
    np.zeros(trials[cl1].shape[2], dtype=int),
    np.ones(trials[cl2].shape[2], dtype=int)
])

# Some information about the dimensionality of the data (channels x time x trials)
print('Shape of trials[cl1]:', trials[cl1].shape)
print('Shape of trials[cl2]:', trials[cl2].shape)




def psd(trials, sfreq, nperseg=None):
    '''
    Calcola la densità spettrale di potenza (PSD) per ciascun canale e trial.

    Parameters
    ----------
    trials : ndarray (channels x samples x trials)
        Segnale EEG.
    sfreq : float
        Frequenza di campionamento (Hz).
    nperseg : int, optional
        Numero di punti per segmento della FFT (default: nsamples).

    Returns
    -------
    trials_psd : ndarray (channels x freqs x trials)
        PSD per ciascun canale e trial.
    freqs : ndarray
        Frequenze associate alla PSD.
    '''
    nchannels, nsamples, ntrials = trials.shape
    if nperseg is None:
        nperseg = nsamples

    # Esegui la PSD trial per trial
    for trial_idx in range(ntrials):
        if trial_idx == 0:
            # Solo al primo ciclo: inizializza PSD di forma corretta
            f, Pxx = welch(trials[:, :, trial_idx], fs=sfreq, nperseg=nperseg, axis=1)
            trials_psd = np.zeros((nchannels, len(f), ntrials))
        else:
            _, Pxx = welch(trials[:, :, trial_idx], fs=sfreq, nperseg=nperseg, axis=1)

        trials_psd[:, :, trial_idx] = Pxx

    return trials_psd, f


# Applica la funzione
sfreq = raw_combined.info['sfreq']
psd_r, freqs = psd(trials[cl1], sfreq)
psd_f, freqs = psd(trials[cl2], sfreq)

trials_PSD = {cl1: psd_r, cl2: psd_f}





def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None):
    '''
    Plotta i dati PSD calcolati con psd().

    Parametri
    ----------
    trials_PSD : dict di array 3D
        I dati PSD, come restituiti da psd(), per ogni classe.
    freqs : lista di float
        Le frequenze per cui è definita la PSD.
    chan_ind : lista di interi
        Gli indici dei canali da plottare.
    chan_lab : lista di stringhe
        (opzionale) Nomi da assegnare a ogni canale nel titolo.
    '''
    plt.figure(figsize=(12, 5))

    nchans = len(chan_ind)
    nrows = int(np.ceil(nchans / 3))
    ncols = min(3, nchans)

    for i, ch in enumerate(chan_ind):
        plt.subplot(nrows, ncols, i + 1)

        for cl in trials_PSD.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch, :, :], axis=1), label=cl)

        plt.xlim(3, 25)

        plt.grid()
        plt.xlabel('Frequenza (Hz)')

        if chan_lab is None:
            plt.title(f'Canale {ch + 1}')
        else:
            plt.title(chan_lab[i])

        plt.legend()

    plt.tight_layout()



plot_psd(
    trials_PSD,
    freqs,
    [channel_names.index(ch) for ch in ['EEG:C3',"EEG:Cz", 'EEG:C4']],
    chan_lab=['sinistra','centro', 'destra'],  # o 'C3', 'C4' se preferisci
)
plt.suptitle("plot PSD", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()






# Filtro Anti-Drift (passa-alto a 0.5 Hz):
def highpass(trials, cutoff, sample_rate, order=3):
    b, a = scipy.signal.butter(order, cutoff / (sample_rate / 2.0), btype='high')
    nchannels, nsamples, ntrials = trials.shape
    trials_hp = np.zeros_like(trials)
    for i in range(ntrials):
        trials_hp[:, :, i] = scipy.signal.filtfilt(b, a, trials[:, :, i], axis=1)
    return trials_hp

#Filtro Notch (per il rumore a 50 Hz):
def notch(trials, cutoff, Q, sample_rate):
    b, a = scipy.signal.iirnotch(w0=cutoff / (sample_rate / 2.0), Q=Q)
    nchannels, nsamples, ntrials = trials.shape
    trials_notched = np.zeros_like(trials)
    for i in range(ntrials):
        trials_notched[:, :, i] = scipy.signal.filtfilt(b, a, trials[:, :, i], axis=1)
    return trials_notched


def bandpass(trials, lowcut, highcut, fs):
    """
    Applica un filtro bandpass ai dati EEG.

    Parameters:
    - trials: array di forma (canali, campioni, prove)
    - lowcut: frequenza di taglio inferiore (in Hz)
    - highcut: frequenza di taglio superiore (in Hz)
    - fs: frequenza di campionamento (in Hz)

    Returns:
    - trials_filt: array dei dati EEG filtrati
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design del filtro bandpass
    b, a = signal.butter(4, [low, high], btype='band')

    # Applicare il filtro a ciascun trial
    trials_filt = np.zeros_like(trials)
    for i in range(trials.shape[2]):
        for j in range(trials.shape[0]):  # canali
            trials_filt[j, :, i] = signal.filtfilt(b, a, trials[j, :, i])

    return trials_filt
# 1. Rimuovi il drift a bassa frequenza
trials_hp = {
    cl1: highpass(trials[cl1], cutoff=0.5 , sample_rate= sfreq),
    cl2: highpass(trials[cl2], cutoff=0.5 , sample_rate=sfreq)
}
# 2. Applica il filtro Notch per rimuovere il rumore elettrico
trials_notch = {
    cl1: notch(trials_hp[cl1], cutoff=50.0 , Q=30.0    , sample_rate=sfreq),
    cl2: notch(trials_hp[cl2], cutoff=50.0 , Q=30.0    , sample_rate=sfreq)
}
# 3. Applica il filtro passa banda (μ e β band)
trials_filt = {
    cl1: bandpass(trials_notch[cl1], 8, 15, sfreq),
    cl2: bandpass(trials_notch[cl2], 8, 15, sfreq)
}


# Plotting the PSD of the resulting `trials_filt` shows the suppression of frequencies outside the passband of the filter:
psd_r, freqs = psd(trials_filt[cl1],sfreq)
psd_f, freqs = psd(trials_filt[cl2],sfreq)
trials_PSD = {cl1: psd_r, cl2: psd_f}
plot_psd(
    trials_PSD,
    freqs,
    [channel_names.index(ch) for ch in ['EEG:C3','EEG:Cz', 'EEG:C4']],
    chan_lab=['sinistra', 'centro','destra'],
)
plt.suptitle("Plot filtered ", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Come caratteristica per il classificatore, useremo il logaritmo della varianza di ciascun canale. La funzione seguente lo calcola


def logvar(trials):
    '''
    Calcola la log-varianza per ciascun trial in un dataset EEG.

    trials: numpy array
        Array di forma (n_canali, n_samples, n_trials) dove n_canali è il numero di canali,
        n_samples è il numero di campioni per trial, e n_trials è il numero di trial.

    Return: numpy array
        Array di log-varianza per ogni trial.
    '''
    # Inizializzare una lista per salvare i valori di log-varianza
    log_var_trials = []

    # Calcolare la log-varianza per ogni trial
    for i in range(trials.shape[2]):  # Itera su tutti i trial
        trial = trials[:, :, i]  # Estrai il trial i-esimo

        # Calcolare la varianza per ogni canale del trial
        var_trial = np.var(trial, axis=1)  # Calcola la varianza lungo il tempo per ogni canale

        # Applicare il logaritmo alla varianza
        log_var_trial = np.log(var_trial + 1e-6)  # Aggiungi un piccolo valore per evitare log(0)

        log_var_trials.append(log_var_trial)

    # Restituisce un array numpy di log-varianza per ogni trial
    return np.array(log_var_trials).T  # La forma finale è (n_trials, n_canali)


# Apply the function
trials_logvar = {cl1: logvar(trials_filt[cl1]),
                 cl2: logvar(trials_filt[cl2])}


def plot_logvar(trials):
    '''
    Plots the log-var of each channel/component.
    arguments:
        trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    '''
    plt.figure(figsize=(12, 5))

    x0 = np.arange(nchannels)
    x1 = np.arange(nchannels) + 0.3

    y0 = np.mean(trials[cl1], axis=1)
    y1 = np.mean(trials[cl2], axis=1)

    plt.bar(x0, y0, width=0.3)
    plt.bar(x1, y1, width=0.3)

    plt.gca().yaxis.grid(True)
    plt.title('log-var of each channel/component')
    plt.xlabel('channels/components')
    plt.ylabel('log-var')
    plt.legend(cl_lab)



# Plot the log-vars
plot_logvar(trials_logvar)
plt.suptitle("Plot logvar", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


def cov(trials):
    '''Calcola la media delle matrici di covarianza normalizzate per ciascun trial.'''
    ntrials = trials.shape[2]
    covs = [trials[:, :, i].dot(trials[:, :, i].T) / nsamples for i in range(ntrials)]
    return np.mean(covs, axis=0)


def whitening(sigma, eps=1e-10):
    '''
    Restituisce una matrice di whitening per la matrice di covarianza data.
    La whitening riduce la correlazione tra i canali EEG.
    '''
    U, s, _ = np.linalg.svd(sigma)
    s_invroot = 1. / np.sqrt(s + eps)  # Stabilizzazione numerica
    return U @ np.diag(s_invroot) @ U.T


def csp(trials_r, trials_f):
    '''
    Hai segnali EEG etichettati per due classi (es. immaginazione mano destra e mano sinistra).

    CSP calcola matrici di covarianza per ogni classe.

    Risolve un problema di autovalori per trovare una matrice di proiezione (filtri spaziali).

    Applica questi filtri ai segnali originali.

    I segnali proiettati vengono passati a un classificatore, spesso dopo aver calcolato la log-varianza (log-var) di ciascun canale filtrato.
    '''
    cov_r = cov(trials_r)
    cov_f = cov(trials_f)
    P = whitening(cov_r + cov_f)
    B, _, _ = linalg.svd(P.T.dot(cov_f).dot(P))
    W = P.dot(B)
    return W


def apply_mix(W, trials):
    ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
    ntrials = trials.shape[2]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:, :, i] = W.T.dot(trials[:, :, i])
    return trials_csp



# Apply the functions
W = csp(trials_filt[cl1], trials_filt[cl2])
print('CSP transformation matrix W:', W)

trials_csp = {cl1: apply_mix(W, trials_filt[cl1]),
              cl2: apply_mix(W, trials_filt[cl2])}

trials_logvar = {cl1: logvar(trials_csp[cl1]),
                 cl2: logvar(trials_csp[cl2])}
plot_logvar(trials_logvar)
plt.suptitle("Plot post applicazioni CSP ", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
psd_r, freqs = psd(trials_csp[cl1],sfreq)
psd_f, freqs = psd(trials_csp[cl2],sfreq)
trials_PSD = {cl1: psd_r, cl2: psd_f}

plot_psd(
    trials_PSD,
    freqs,
    [channel_names.index(ch) for ch in ['EEG:C3',"EEG:Cz",'EEG:C4']],
    chan_lab=['sinistra',"centro", 'destra'],  # o 'C3', 'C4' se preferisci,
     )
plt.show()


#  Training a classifier
def plot_scatter(left, right):
    plt.figure()
    plt.scatter(left[0,:], left[-1,:])
    plt.scatter(right[0,:], right[-1,:])
    plt.xlabel('left')
    plt.ylabel('right')
    plt.legend(cl_lab)
plot_scatter(trials_logvar[cl1], trials_logvar[cl2])
plt.show()


n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Unisci i dati delle due classi
X_all = np.concatenate((trials_filt[cl1], trials_filt[cl2]), axis=2)  # shape: (n_channels, n_samples, n_trials)
y_all = np.concatenate((
    np.ones(trials_filt[cl1].shape[2]),
    np.full(trials_filt[cl2].shape[2], 2)
))

accuracies = []
all_confusions = np.zeros((2, 2))
y_true_global = []
y_pred_global = []
fold = 1

for train_idx, test_idx in skf.split(np.zeros(len(y_all)), y_all):
    print(f"Fold {fold}...")

    # Suddividi i trial in base agli indici dei fold
    train_trials = X_all[:, :, train_idx]
    test_trials = X_all[:, :, test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    # Ricostruisci i dizionari train/test per ciascuna classe
    train = {
        cl1: train_trials[:, :, y_train == 1],
        cl2: train_trials[:, :, y_train == 2]
    }
    test = {
        cl1: test_trials[:, :, y_test == 1],
        cl2: test_trials[:, :, y_test == 2]
    }

    # CSP
    W = csp(train[cl1], train[cl2])
    for label in [cl1, cl2]:
        train[label] = apply_mix(W, train[label])
        test[label] = apply_mix(W, test[label])

    # Componenti selezionate
    comp = np.array([0, -1])
    for label in [cl1, cl2]:
        train[label] = train[label][comp, :, :]
        test[label] = test[label][comp, :, :]
        train[label] = logvar(train[label])
        test[label] = logvar(test[label])

    # Prepara X e y per l'SVM
    X_train = np.concatenate((train[cl1].T, train[cl2].T), axis=0)
    y_train = np.concatenate((
        np.ones(train[cl1].shape[1]),
        np.full(train[cl2].shape[1], 2)
    ))
    X_test = np.concatenate((test[cl1].T, test[cl2].T), axis=0)
    y_test = np.concatenate((
        np.ones(test[cl1].shape[1]),
        np.full(test[cl2].shape[1], 2)
    ))
    # SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm = SVC(kernel='linear')

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # Metriche
    conf = confusion_matrix(y_test, y_pred, labels=[1, 2])
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    all_confusions += conf
    y_true_global.extend(y_test.tolist())
    y_pred_global.extend(y_pred.tolist())

    # Visualizzare i risultati di ciascun fold
    print(f"Fold {fold} - Accuracy: {acc:.3f}")
    print(f"Confusion Matrix (Fold {fold}):")
    print(conf)

    plot_scatter(train[cl1], train[cl2])
    plt.title(f'Training data and Decision Boundary (Fold {fold})')
    w = svm.coef_[0]
    b = svm.intercept_[0]
    x = np.arange(-5, 1, 0.1)
    y = -(w[0] * x + b) / w[1]
    plt.plot(x, y, 'k--')
    plt.xlim(-3, 2)
    plt.ylim(-2.5, 2)
    plot_scatter(test[cl1], test[cl2])
    plt.title(f'Test set (Fold {fold})')
    # Visualizzare la frontiera di decisione
    w = svm.coef_[0]
    b = svm.intercept_[0]
    x = np.arange(-5, 1, 0.1)
    y = -(w[0] * x + b) / w[1]
    plt.plot(x, y, 'k--')
    plt.xlim(-3, 2)
    plt.ylim(-2.5, 2)
    #plt.show()

    fold += 1  # Incrementa il contatore del fold
# Calcolare l'accuratezza media
mean_accuracy = np.mean(accuracies)

# Calcolo del valore kappa totale
kappa_total = cohen_kappa_score(y_true_global, y_pred_global)

# Visualizzare la matrice di confusione totale e le metriche finali
print("Confusion Matrix Totale:")
print(all_confusions.astype(int))
print(f"Accuracy media (su {n_splits} fold): {mean_accuracy:.3f}")
print(f"Kappa totale: {kappa_total:.3f}")