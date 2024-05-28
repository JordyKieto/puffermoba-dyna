FROM pufferai/puffer-deps:1.0

RUN mkdir -p /puffertank
WORKDIR /puffertank

RUN git clone https://github.com/facebookresearch/nle --recursive && pip3 install --no-dependencies -e nle/.
RUN git clone https://github.com/pufferai/pufferlib --branch 1.0 && pip3 install --user -e pufferlib/[cleanrl,atari]

COPY version_check.py /root/version_check.py
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

# Copy my personal NeoVim config
COPY init.vim /root/.config/nvim/init.vim

# For the memes. Properly escaped pufferfish prompt
RUN echo "export PS1=$' \xf0\x9f\[\x90\xa1\] '" >> ~/.bashrc \
 && echo "alias vim='/usr/bin/nvim'" >> ~/.bashrc

ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["/bin/bash"]
