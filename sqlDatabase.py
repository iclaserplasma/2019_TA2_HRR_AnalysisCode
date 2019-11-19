from mysql.connector import (connection)
#import mysql.connector


def connectToSQL(connectionDetails,debug):
	''' Create a connection object for the SQL database
	the connection details are stored in a file called connectionDetails
	'''
	f = open(connectionDetails, "r")

	user = f.readline()[0:-1]
	password = f.readline()[0:-1]
	host = f.readline()[0:-1]
	database = f.readline()

	if debug:
		print('User: ' + user  + '\nPassword: ' + password + '\nhost: ' + host + '\ndatabase: ' + database)

	try:
		cnx = connection.MySQLConnection(user=user, password=password,host=host, database=database)
		#cnx = mysql.connector.connect(user=user, password=password,host=host, database=database)
	except:
		print('[ERROR] Cannot make a connection to the SQL server')

	return cnx
