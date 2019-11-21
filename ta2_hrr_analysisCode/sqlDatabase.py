from mysql.connector import (connection)
#import mysql.connector

def connectToSQL(debug):
	''' Create a connection object for the SQL database
	'''

	user = 'iclaserp_ta2user'
	password = r'[GV]user~000'
	host = 'iclaserplasmadbs.com'
	database = 'iclaserp_Streeter2019'

	if debug:
		print('User: ' + user  + '\nPassword: ' + password + '\nhost: ' + host + '\ndatabase: ' + database)

	try:
		cnx = connection.MySQLConnection(user=user, password=password,host=host, database=database)
		#cnx = mysql.connector.connect(user=user, password=password,host=host, database=database)
	except:
		print('[ERROR] Cannot make a connection to the SQL server')

	return cnx
