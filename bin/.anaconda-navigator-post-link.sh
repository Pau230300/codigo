#!/bin/bash

# Remove old Navigator.app if it exists
#rm -rf $PREFIX/Navigator.app
#rm -rf /Applications/Navigator.app
#rm -rf $HOME/Applications/Navigator.app
#rm -rf /Applications/Anaconda-Navigator.app
#rm -rf $HOME/Applications/Anaconda-Navigator.app

# Renaming the app folder must happen prior to signing
# so this is now in build.sh
#cp -r $PREFIX/navigatorapp $PREFIX/Anaconda-Navigator.app
#rm -rf $PREFIX/navigatorapp

ln -s -f $PREFIX/Anaconda-Navigator.app /Applications/ >/dev/null 2>&1
if (( $? )); then
    mkdir -p $HOME/Applications
    ln -s -f $PREFIX/Anaconda-Navigator.app $HOME/Applications/ || exit 0
fi
