# An-Automated-Traditional-Chinese-Dialogue-Generating-System
*Generating conversation dialogs from Traditional-Chinese dataset using Hierarchical Neural Network and Attention Seq2seq*

- Implement with Teacher Forcing and Scheduled Sampling
- Using Pytorch

## Usage

Hierarchical Neural Network
```
python main.py --data=examples/example --save=model/
```

Attention Seq2seq
```
python seq2seq.py --data=examples/example --save=model/
```

Gernerating Samples
```
python gen.py --data=examples/example --save=model/ --type=[hrnn or seq2seq]
```
