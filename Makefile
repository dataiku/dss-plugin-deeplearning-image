PLUGIN_VERSION=2.0.0
PLUGIN_ID=deeplearning-image-v2

plugin:
	cat plugin.json|json_pp > /dev/null
	rm -rf dist
	mkdir dist
	zip -r dist/dss-plugin-${PLUGIN_ID}-${PLUGIN_VERSION}.zip code-env custom-recipes js plugin.json python-lib python-runnables resource webapps

include ../Makefile.inc