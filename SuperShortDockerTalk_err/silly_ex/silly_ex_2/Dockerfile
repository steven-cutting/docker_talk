

FROM sillyex:1
MAINTAINER Steven Cutting <steven.e.cutting@linux.com>

ENV TESTLIST $TESTLIST:monkey_three
RUN echo $TESTLIST

# A little hack... Just go with it.
RUN echo "echo $TESTLIST" > test.sh && chmod +x test.sh
CMD ["/bin/bash", "./test.sh"]
