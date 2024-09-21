base_file=./src/bin/$1.rs
submit_file=./src/bin/submit.rs
rm -f $submit_file

cat $base_file | grep -v mod > $submit_file
cat $base_file | grep mod | sed -E 's/mod (.*);(.*)/\1/' > mod.txt

items=(`cat mod.txt`)
for item in "${items[@]}" ; do
    echo -e "\n" >> $submit_file
    if [ $item = "vis" ]; then
        echo '#[cfg(feature = "local")]' >> $submit_file
    fi
    echo "mod $item {" >> $submit_file
    cat ./src/bin/$item.rs >> $submit_file
    echo "}" >> $submit_file
done

rustfmt $submit_file
rm mod.txt

cargo run -r --bin submit --features local