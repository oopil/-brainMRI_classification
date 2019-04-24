import os, subprocess, time

class Printer():
	def __init__(self):
		pass
	
	def p_highlight(self, line):
		line_ = '-'*50
		print(line_)
		print(line)
		print(line_)

	def p_list(self, l):
		for line in l:
			print(line)

	def p_list_index(self, l):
		for line in l:
			pass
			
#%%
def print_by_i(string,i):
	 #assert string != None
	if string == None:
		print('the string is NoneType in print_by_i')
		return
	tmp = string.split('/')
	print(tmp[i])

def print_l(l):
	for line in sorted(l):
		print(line)

def print_l_by_i(l, index):
	set_a = []
	for line in sorted(l):
		name = line.split('/')
		set_a.append(name[index])
		print(name[index])
	return set(set_a)
#%%
class System():
	def __init__(self, order, options):
		self.order = order
		self.options = options
		self.command = order +' '+ options

	def sub_run(self):
		print(self.command)
		program = self.command.split()[0]
		process = subprocess.Popen(self.command.split())
		time.sleep(0.1)
		process.wait()
		print("{} process ends".format(program))

	def run(self):
		print(self.command)
		program = self.command.split()[0]
		os.system(self.command)
		print("{} process ends".format(program))
#%%

class Dcm2nii():
	def __init__(self, path):
		self.dcm2nii = self.set_dcm2nii_path(path)
		pass
	
	def set_dcm2nii_path(self, path):
		self.dcm2nii_path = path

	def set_fld_list(self, fld_list):
		 # set path list that exists dicom files to convert
		 self.fld_list = fld_list
	
	def convert_all_fld(self):
		for fld in self.fld_list:
			self.convert(fld)

	def convert(self, fld):
		 #command = '.'+self.dcm2nii_path+' '+fld+'/*'
		sys = System(self.dcm2nii_path, fld+'/*')
		sys.run()
#%%
class FileRemover():
	def __init__(self):
		pass
	
	def set_fld_list(self, fld_list):
		self.fld_list = fld_list
	
	def set_rm_list(self, rm_list):
		self.rm_list = rm_list
	
	def remove_all(self, fld_path, sub_str):
		sys = System('rm', fld_path+'/*'+sub_str+'*')
		sys.run()
	
	def remove_options(self):
		for fld in self.fld_list:
			for option in self.rm_list:
				self.remove_all(fld, option)
		
#%%
class MetaBot():
	def __init__(self, path):
		self.set_base_fld(path)
		self.fld_list = []
		self.image_list = []
		self.img_l_mv = []
		self.type = ['T1_DSPGR','T2','DTI','fMRI']
	
	def get_fld_by_type(self, fld_type):
		for fld_name in self.type:
			if fld_type in fld_name:
				return fld_type
		print('there is no appropriate folder type in get_fld_by_type.')
		return 'None'

	def set_base_fld(self, path):
		self.base_path = path

		#file all fld path that contain dicom file
	def find_all_fld(self, path):
		file_list = self.get_file_list(path)
		for file_name in file_list:
			# check the dicom file exist in fld
			# whether the 1st letter is I
#			prs_l = path.split('/')
#			fld_name = prs_l[-1]
			new_path = self.join_path(path,file_name)
			#if self.is_1st_letter_sm(self.letter_option, file_name) and self.is_dir(new_path) == False:
			if 'I00' in file_name:
				self.fld_list.append(path)
				break
	
			if self.is_dir(new_path):
				self.find_all_fld(new_path)
	
	def get_fld_list(self):
		return self.fld_list
	
	def set_options(self, word_list):
		self.options = word_list

	def find_all_img_path(self, path, substr):
		file_list = self.get_file_list(path)
		for file_name in file_list:
			prs_l = path.split('/')
			fld_name = prs_l[-1]
			new_path = self.join_path(path,file_name)
			if self.is_dir(new_path):
				self.find_all_img_path(new_path, substr)
				continue

			if self.is_contain_substr(file_name, substr):
				self.img_l_mv.append(new_path)

	def find_all_image(self, fld_list, i_type):
		self.no_dti = []
		for fld in fld_list:
			fld_type = self.get_fld_by_type(i_type)
			fld = fld +'/'+ fld_type
			file_path = self.find_image_in_fld(fld, i_type)
			if type(file_path) == None: 
				print('Got a None type file path')
				continue
			self.image_list.append(file_path)
			 #assert False

	def find_image_in_fld(self, fld, i_type):
		img_l = self.get_file_list(fld)
		tmp_list = []
		for image in img_l:
			new_path = self.join_path(fld, image)
			if self.is_option(i_type, image):
				tmp_list.append(new_path)
		if self.no_image(tmp_list):
			self.no_dti.append(fld)
			print('there is no image in the fld {}'.format(fld))
			return 'None'

		if self.one_image(tmp_list):
			return tmp_list[0]

		#if there are images more than one, we have to choose
		if i_type == 'T1':
			return self.sel_T1(tmp_list)
		elif i_type == 'DTI':
			return self.sel_DTI(tmp_list)
#		elif i_type == 'T2':
#			return self.T2_op(name)
#		elif i_type == 'fMRI':
#			return self.fMRI_op(name)
		else:
			print('image type is not valid : {}',format(i_type))

	def rm_all_sm_file(self, fld_list):
		for fld in fld_list:
			self.rm_sm_file_in_fld(fld)

	def rm_sm_file_in_fld(self, fld_path):
		tmp_list = self.get_file_list(fld_path)
		new = ''
		old = ''
		for image in sorted(tmp_list):
			old = new
			new = file_name
			self.rm_sm_file(old, new)

	def rm_sm_file(self, f1, f2):
		remover = FileRemover()
		if self.gz_clone(f1,f2) or self.f_clone(f1,f2):
			print('there are files with same name and diff extensions')
			remover.remove(f2)
		elif self.gz_clone(f2,f1) or self.f_clone(f2,f1):
			print('there are files with same name and diff extensions')
			remover.remove(f1)
		del remover
	
	def f_clone(self, f1, f2):
		if 'f'+f1 ==f2: return True
		else: return False

	def gz_clone(self, f1, f2):
		if f1+'.gz'==f2: return True
		else: return False

	def get_image_l_w_substr(self,l, substr):
		files = []
		for elem in sorted(l):
			if self.is_contain_substr(elem, substr):
				files.append(elem)
		return files

	def is_exist(self, path):
		return os.path.exists(path)

	def is_contain_substr(self, name, substr):
		if substr in name:
			return True
		return False

	def join_path(self, path, name):
		return path + '/' + name
	
	def find_no_DTI_fld(self, dti_img_l):
		find_list = []
		for img_set in dti_img_l:
			for f in img_set:
				if f == 'None':
					find_list.append(img_set)
					break
		return find_list

	def is_1st_letter_sm(self, letter, name):
		if name[0] == str(letter) and len(name) > 1: return True
		else: False

	def is_file_options(self, options, name):
		for option in options:
			if option in name:
				return True
		return False
	
	def is_dir(self, path):
		return os.path.isdir(path)
	
	def get_all_image_count(self):
		return self.all_image_count

	def get_file_list(self, directory):
		try:
			image_list = os.listdir(directory)
		except FileNotFoundError as e:
			print(e)
			image_list = []
		return image_list
		
	def is_image(self, row):
		if row[2] == 'O':
			return True
		else:
			self.no_image_people.append(row[1])
			return False

	def option(self, name):
		 # we use the files does not start with c or co
		if 'brain' in name:
			return False

		if name[0] == str('c') or name[0] == str('o'):
			return False
		else:
			return True

	def extr_name(self, path):
		last = len(path) -1
		for index in range(last):
			if path[last-index] == '/':
				return path[last-index+1:]

	def get_name_list(self):
		self.name_list = []
		for row in self.data_list:
			print(row)
			if row[1] is not None and len(row[1]) > 2:
				self.name_list.append(row[1])
		return self.name_list
			
#%%
class FileWriter():
	def __init__(self, data, fld_path, result_name):
		self.data = data
		self.fld_path = fld_path
		self.result_name = result_name
	
	def open_file(self):
		self.result_file_name = self.fld_path + self.result_name + '.txt'
		self.fd = open(self.result_file_name, 'w+')
	
	def write_data(self):
		for line in self.data:
			self.fd.write(self.merge_line(line))
			self.fd.write('\n')
	
	def merge_line(self, line):
		merged_line = ''
		for word in line:
			merged_line = merged_line + ' ' + word
		return merged_line
#%%
