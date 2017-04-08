JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	CNN.java \
	Instance.java \
	Dataset.java \
	ANN.java \
	BoundMath.java \
	BoundNumbers.java \
	Mat2D.java \
	Mat3D.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class