
import os, sys, time

iter_count = 1

app_list = []
main_folder = "/home/mghane/gecko-microbench/ACC"

if len(sys.argv)>1:
	app_list.append(sys.argv[1])
else:
	app_list = ['vector_add', 'stencil']



for app_name in app_list:
	dev_list = [1, 2, 3, 4]
	if app_name == "gaussian" or app_name == "lavaMD":
		dev_list = [4]
	for dev_count in dev_list:
		if app_name == "gaussian" and dev_count in [1,2,3]:
			continue
		os.environ["GECKO_CONFIG_FILE"] = "/home/mghane/gecko-rodinia/config/gecko_%d.conf" % (dev_count)
		print "Setting device count to %d for %s" % (dev_count, app_name)
		os.chdir(main_folder+app_name)
		for i in range(iter_count):
			result_file = "%s-%d-gecko-gpu-trace-ACC.txt" % (app_name, dev_count)
			print "at iteration %d" % (i)
			cmd = "nvprof --profile-child-processes --print-gpu-trace sh run &> %s" % (result_file)
			os.system(cmd)
			time.sleep(3)


