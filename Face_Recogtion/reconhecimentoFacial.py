import cv2
import os
import time
# from pyedo import eduedo as edo # Pyedo for real robot Control
from edoSim import edosim as eduedo # EdoSim for the Simulated Control

dataPath = "C:/Users/lsv/Downloads/VerificacaoFacialRoboTerminalCoppellia/Data" # Mudar a rota onde esta armazenada a data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()


# Lendo o modelo
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rosto = auxFrame[y:y+h,x:x+w]
		rosto = cv2.resize(rosto,(150,150),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(rosto)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

		# LBPHFace
		if result[1] < 70:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Desconhecido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

	cv2.imshow('frame',frame)
	# k = cv2.waitKey(1)
	# if k == 27:
	# 	break
	if cv2.waitKey(1) & 0xff == ord('k'):

		cap.release()
		cv2.destroyAllWindows()

		edo = eduedo('127.0.0.1')

		# Init Axes
		edo.init7Axes()

		# Move to Default Position
		edo.moveToHome()

		# Start Movement

		iniciar = "s"

		if iniciar == "s":
			edo.moveJoints(6.3376915497198, 8.867252108834, -33.320176280937, 14.851945821869, 25.074867537546, -13.64536558423) #casa
			time.sleep(6)
			edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677) #segura
			time.sleep(6)
			edo.moveJoints(39.269407651801, -57.019826210235, -31.094996271995, 39.182841834261, 88.393412238097, -1.3225158686112) #identificação do mesário
			time.sleep(6)
			print("Identificação do mesário")
			edo.moveJoints(0,0,0,0,0,0) #segura
			time.sleep(6)
			edo.moveJoints(2.8938828494427, -41.719054997626, 5.6820753516686, 4.9037752957457, 36.254835359724, -3.9519884349044) #antes da identificação do eleitor (ok)
			time.sleep(6)
			edo.moveJoints(2.8834736411476, -55.016859575751, 5.0006683593115, 3.8333070678863, 50.199950625869, -2.4818585179509) #identificação do eleitor (ok)
			time.sleep(6)
			print("Identificação do eleitor")
			edo.moveJoints(2.8938828494427, -41.719054997626, 5.6820753516686, 4.9037752957457, 36.254835359724, -3.9519884349044) #voltar para identificação do eleitor (ok)
			time.sleep(6)
			edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677) #segura
			time.sleep(6)


		corrige = "s"
		tecla = 0
		b = 0
		print(result[0])
		if result[0] == 11:
			tituloEleitor = "123456789012"
		while corrige == "s" and iniciar == "s":
			print("Realizando identificação...  ")
			for i in tituloEleitor:
				tecla = int(i)
				if tecla == 0:
					edo.moveJoints(8.5836216733601, -2.4701105925784, -61.247699646095, 9.5712123858306, 64.12938377768, -4.2326272475219) 
					time.sleep(4)
					edo.moveJoints(6.4394477079747, -54.516561879426, -53.8577764736, 6.6848700652316, 108.2994931401, 2.2012197053334) 
					time.sleep(6)
					print("0")
					edo.moveJoints(8.5836216733601, -2.4701105925784, -61.247699646095, 9.5712123858306, 64.12938377768, -4.2326272475219) 
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)
					
				if tecla == 1:
					edo.moveJoints(11.082227815156, -48.818790753044, -31.951925465643, 11.303484963129, 80.913125662882, -1.9154172697034) 
					time.sleep(4)
					edo.moveJoints(10.062958685315, -54.520072596659, -29.143187754394, 10.121725632934, 83.854382704408, -1.1107800043408) 
					time.sleep(2)
					print("1")
					edo.moveJoints(11.082227815156, -48.818790753044, -31.951925465643, 11.303484963129, 80.913125662882, -1.9154172697034) 
					time.sleep(2) 
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)

				if tecla == 2:
					edo.moveJoints(3.951769868851, -48.511473221529, -33.018309240379, 4.0619408201633, 81.513007517306, -0.68452155861868) 
					time.sleep(4)
					edo.moveJoints(3.9059256391418, -55.360445411759, -29.53707110345, 3.9296127351834, 85.002960975616, -0.35465074246374) 
					time.sleep(2)
					print("2")
					edo.moveJoints(3.951769868851, -48.511473221529, -33.018309240379, 4.0619408201633, 81.513007517306, -0.68452155861868) 
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)

				if tecla == 3:
					edo.moveJoints(-4.4130535246369, -48.53080265688, -32.957192707686, -4.3876998624379, 81.475086307034, 0.55444743606496) 
					time.sleep(4)
					edo.moveJoints(-3.261333366411, -55.019291123095, -29.717879871159, -3.2724392540015, 84.838066548674, 0.29308341728525) 
					time.sleep(2)
					print("3")
					edo.moveJoints(-4.4130535246369, -48.53080265688, -32.957192707686, -4.3876998624379, 81.475086307034, 0.55444743606496) 
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)

				if tecla == 4:
					edo.moveJoints(11.926712404136, -46.621505236673, -40.734988662768, 12.017881769178, 87.371684240155, -0.66225514192444) # antes de clicar (não mexer)
					time.sleep(4)
					edo.moveJoints(12.642557210289, -54.122651209613, -36.759053584633, 12.487812444453, 90.923491891588, 0.30338334255363) # clicando (ok)
					time.sleep(2)
					print("4")
					edo.moveJoints(11.926712404136, -46.621505236673, -40.734988662768, 12.017881769178, 87.371684240155, -0.66225514192444) # antes de clicar
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677) #posição segura
					time.sleep(4)

				if tecla == 5:
					edo.moveJoints(4.9572556769469, -46.432677826879, -41.6860242028, 5.0496817967955, 88.080466630705, -0.28248296369339) 
					time.sleep(4)
					edo.moveJoints(5.1046511592357, -52.891618894745, -38.935548005171, 5.1576807479517, 91.908241245364, 0.13510114178285) 
					time.sleep(2)
					print("5")
					edo.moveJoints(4.9572556769469, -46.432677826879, -41.6860242028, 5.0496817967955, 88.080466630705, -0.28248296369339) 
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)

				if tecla == 6:
					edo.moveJoints(-4.3811428808346, -46.429508619104, -41.70676065712, -4.2964895162615, 88.095575009149, 0.023168001664683) 
					time.sleep(4)
					edo.moveJoints(-3.989895984798, -53.292619300916, -38.63107183235, -4.0061245142659, 92.017264724896, -0.10816287569632) 
					time.sleep(2)
					print("6")
					edo.moveJoints(-4.3811428808346, -46.429508619104, -41.70676065712, -4.2964895162615, 88.095575009149, 0.023168001664683) 
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)

				if tecla == 7:
					edo.moveJoints(13.862374354539, -45.503949684676, -47.856512721638, 13.978337306268, 93.206906093393, 0.69560012545244) 
					time.sleep(4)
					edo.moveJoints(14.306650839292, -53.153215139956, -44.380929981342, 14.457816586003, 97.394112582935, 1.8209967346171) 
					time.sleep(2)
					print("7")
					edo.moveJoints(13.862374354539, -45.503949684676, -47.856512721638, 13.978337306268, 93.206906093393, 0.69560012545244) 
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)

				if tecla == 8:
					edo.moveJoints(3.68103483053, -43.834760734551, -47.688995502054, 3.7508120430908, 91.621988017249, 0.039410191510974) 
					time.sleep(4)
					edo.moveJoints(5.0038785482214, -52.897574819701, -45.159913056182, 5.0885319127945, 98.112757766647, 0.67001423682157) 
					time.sleep(2)
					print("8")
					edo.moveJoints(3.68103483053, -43.834760734551, -47.688995502054, 3.7508120430908, 91.621988017249, 0.039410191510974) 
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)

				if tecla == 9:
					edo.moveJoints(-4.674431203795, -44.084636375147, -49.286344522489, -4.6560443345493, 93.457041281178, -0.33439240138548) 
					time.sleep(4)
					edo.moveJoints(-5.2009021850196, -53.236762013884, -46.706117620111, -5.2279497341328, 100.00160560048, -1.049963999971) 
					time.sleep(2)
					print("9")
					edo.moveJoints(-4.674431203795, -44.084636375147, -49.286344522489, -4.6560443345493, 93.457041281178, -0.33439240138548) 
					time.sleep(2)
					edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677)  
					time.sleep(4)
				b = b + 1
				
				if b == len(tituloEleitor):
					corrige = input("deseja corrigir? s/n  ")
					b = 0
					if corrige == "s":
						edo.moveJoints(19.541772272053, -8.3722136581698, -59.693585723107, 20.955471166084, 69.479757822496, -7.6385420582838) 
						time.sleep(6)
						edo.moveJoints(19.501323891789, -53.848897227679, -51.179372432093, 20.187785224132, 104.22228001695, 5.1323680668877) 
						time.sleep(4)
						print("Corrige")
						edo.moveJoints(19.541772272053, -8.3722136581698, -59.693585723107, 20.955471166084, 69.479757822496, -7.6385420582838) 
						time.sleep(6)
						edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677) 
						time.sleep(10)
								
					if corrige == "n":
						edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677) #segura
						time.sleep(6)
						edo.moveJoints(-9.0205215538091, -17.967632234419, -61.576218084794, -9.1230290328727, 79.761774334872, 1.6434937784668) #confirmaseguro
						time.sleep(6)
						edo.moveJoints(-9.5989975953742, -53.56917366041, -53.298097112631, -9.9596042627942, 106.71691098865, -2.7970580877686) #confirma
						time.sleep(2)
						print("Confirma")
						print("Identificação realizada, obrigado!")
						edo.moveJoints(-9.0205215538091, -17.967632234419, -61.576218084794, -9.1230290328727, 79.761774334872, 1.6434937784668) #confirmaseguro
						time.sleep(2)
						edo.moveJoints(5.8850685738014, -11.013502451727, -42.66819174507, 7.217201348764, 53.740310880254, -4.4078625808677) #segura
						time.sleep(6)
						edo.moveJoints(6.3376915497198, 8.867252108834, -33.320176280937, 14.851945821869, 25.074867537546, -13.64536558423) #casa
						time.sleep(12)

		break