#!/usr/bin/env bash

declare -a trainclasses=("n02795169" "n04404412" "n04251144" "n03476684" "n01910747" "n02504013" "n04509417" "n03062245" "n03207743" "n02701002" "n03124170" "n03527444" "n04254777" "n03337140" "n02111277" "n03017168" "n02108551" "n02115641" "n01749939" "n12345899" "n02823428" "n03530642" "n01784675" "n03676483" "n01963571" "n03888605" "n04243546" "n03355925" "n11943660" "n09246464" "n11904109" "n02165456" "n01773157" "n02277742" "n04141975" "n13054560" "n04275548" "n07747607" "n02687172" "n04552348" "n02108089" "n03400231" "n02120079" "n02137549" "n02898711" "n04443257" "n03447447" "n02165105" "n02096294" "n04496726" "n03924679" "n02101006" "n07730406" "n03047690" "n04435653" "n02966193" "n02747177" "n02200198" "n02325366" "n01909906" "n04596742" "n04515003" "n04389033" "n03220513" )
declare -a testclasses=("n01770393" "n01981276" "n03146219" "n04418357" "n03544143" "n09421951" "n02099601" "n03272010" "n02219486" "n03127925" "n12301445" "n02116738" "n02129165" "n01930112" "n02110341" "n06267145" "n04522168" "n04146614" "n02443484" "n02871525" )
declare -a valclasses=("n02793495" "n04584207" "n12302071" "n03838899" "n01770081" "n02113712" "n13133613" "n02114548" "n04258138" "n03347037" "n03075370" "n12754981" "n01796340" "n07735510" "n03452741" "n04417672" )

# you can register an account on the imagenet website.
token=<accesskey>
user=<username>
DATADIR=data/mini-imagenet

# Create directories
mkdir -p $DATADIR/data
mkdir -p $DATADIR/data/train
mkdir -p $DATADIR/data/test

# Download train set
for i in "${trainclasses[@]}"
do
   wget -c --output-document=$DATADIR/data/train/$i.tar "http://www.image-net.org/download/synset?wnid=$i&username=$user&accesskey=$token&release=latest&src=stanford"
   mkdir $DATADIR/data/train/$i
   tar xf $DATADIR/data/train/$i.tar -C $DATADIR/data/train/$i/
done

# Download test set
for j in "${testclasses[@]}"
do
   wget -c --output-document=$DATADIR/data/test/$j.tar "http://www.image-net.org/download/synset?wnid=$j&username=$user&accesskey=$token&release=latest&src=stanford"
   mkdir $DATADIR/data/test/$j
   tar xf $DATADIR/data/test/$j.tar -C $DATADIR/data/test/$j/
done


# Download val set
for j in "${valclasses[@]}"
do
   wget -c --output-document=$DATADIR/data/val/$j.tar "http://www.image-net.org/download/synset?wnid=$j&username=$user&accesskey=$token&release=latest&src=stanford"
   mkdir $DATADIR/data/val/$j
   tar xf $DATADIR/data/val/$j.tar -C $DATADIR/data/val/$j/
done